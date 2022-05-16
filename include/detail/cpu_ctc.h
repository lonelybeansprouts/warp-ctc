#pragma once

#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#if !defined(CTC_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "ctc_helper.h"

#include <iostream>


template<typename ProbT>
class CpuCTC {
public:
    // Noncopyable
    CpuCTC(int alphabet_size, int minibatch, void* workspace, int num_threads) :
            alphabet_size_(alphabet_size), minibatch_(minibatch),
            num_threads_(num_threads), workspace_(workspace){
#if defined(CTC_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    CpuCTC(const CpuCTC&) = delete;
    CpuCTC& operator=(const CpuCTC&) = delete;

    ctcStatus_t cost_and_grad(const ProbT* const activations,
                              ProbT *grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);


    ctcStatus_t score_forward(const ProbT* const activations,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:

    class CpuCTC_metadata {

    private:
        int setup_labels(const int* const labels, int L);

    public:
        CpuCTC_metadata(int L, int T, int mb,
                        int alphabet_size,
                        void* workspace, size_t bytes_used,
                        const int* const labels);

        ProbT* alphas;
        ProbT* betas;
        int* labels_w;
        int* e_inc;
        int* s_inc;
        ProbT* output;
        int repeats;
    };

    int alphabet_size_; // Number of characters plus blank
    int minibatch_;
    int num_threads_;
    void* workspace_;

    void softmax(const ProbT* const activations, ProbT* probs,
                 const int* const input_lengths);

    std::tuple<ProbT, bool>
            cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                 const int* const labels, int T, int L,
                                 int mb, size_t bytes_used);

    ProbT compute_alphas(const ProbT* probs, int L, int T,
                                    const int* const labels,
                                    ProbT* alphas);

    ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                            ProbT log_partition,
                                            int L, int T, 
                                            const int* const labels,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output);
};

template<typename ProbT>
CpuCTC<ProbT>::CpuCTC_metadata::CpuCTC_metadata(int L, int T, int mb,
                                                int alphabet_size,
                                                void* workspace, size_t bytes_used,
                                                const int* const labels) {

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * L * T;
    std::fill(alphas, alphas + L * T, ctc_helper::neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * L;
    std::fill(betas, betas + L, ctc_helper::neg_inf<ProbT>());
    labels_w = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * L;
    output = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * alphabet_size;

    repeats = setup_labels(labels, L);
}

template<typename ProbT>
int CpuCTC<ProbT>::CpuCTC_metadata::setup_labels(const int* const labels, int L) {
    for (int i = 0; i < L; ++i) {
        labels_w[i] = labels[i];
    }
    return 0;
}

template<typename ProbT>
void
CpuCTC<ProbT>::softmax(const ProbT* const activations, ProbT* probs,
                       const int* const input_lengths) {
#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for(int c = 0; c < input_lengths[mb]; ++c) {
            int col_offset = (mb + minibatch_ * c) * alphabet_size_;
            ProbT max_activation = -std::numeric_limits<ProbT>::infinity();
            for(int r = 0; r < alphabet_size_; ++r)
                max_activation = std::max(max_activation, activations[r + col_offset]);

            ProbT denom = ProbT(0.);
            for(int r = 0; r < alphabet_size_; ++r) {
                probs[r + col_offset] = std::exp(activations[r + col_offset] - max_activation);
                denom += probs[r + col_offset];
            }

            for(int r = 0; r < alphabet_size_; ++r) {
                probs[r + col_offset] /= denom;
            }
        }
    }
}

template<typename ProbT>
std::tuple<ProbT, bool>
CpuCTC<ProbT>::cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                    const int* const labels,
                                    int T, int L, int mb, size_t bytes_used) {

    CpuCTC_metadata ctcm(L, T, mb, alphabet_size_, workspace_, bytes_used, labels);

    bool over_threshold = false;

    if (L > T) {
        return std::make_tuple(ProbT(0), over_threshold); // TODO, not right to return 0
    }

    ProbT llForward = compute_alphas(probs, L, T, ctcm.labels_w, ctcm.alphas);
    ProbT llBackward = compute_betas_and_grad(grad, probs, llForward,
                                              L, T, 
                                              ctcm.labels_w,
                                              ctcm.alphas,
                                              ctcm.betas,
                                              ctcm.output);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
        over_threshold = true;
    }
    return std::make_tuple(-llForward, over_threshold);
}

// Computes forward probabilities
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_alphas(const ProbT* probs, int L, int T,
                                    const int* const labels,
                                    ProbT* alphas) {
    //initial
    alphas[0] = std::log(probs[labels[0]]);

    int start = 0, end = 1;
    for(int t = 1; t < T; ++t) {
        int remain = L - (T - t);
        if(remain > 0)
            start ++;
        if(t < L)
            end ++;
        int startloop = start;
        int idx1 = t * L, idx2 = (t - 1) * L, idx3 = t * (alphabet_size_ * minibatch_);

        if (start == 0) {
            alphas[idx1] = alphas[idx2] + std::log(probs[labels[0] + idx3]);  //start=0， no more previous token 
            startloop += 1;
        }

        for(int i = startloop; i < end; ++i) {
            ProbT prev_sum = ctc_helper::log_plus<ProbT>()(alphas[i + idx2], alphas[(i-1) + idx2]);
            alphas[i + idx1] = prev_sum + std::log(probs[labels[i] + idx3]);
        }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglike = ctc_helper::log_plus<ProbT>()(loglike, alphas[i + (T - 1) * L]);
    }

    return loglike;
}

// Starting from T, we sweep backward over the alpha array computing one column
// of betas as we go.  At each position we can update product alpha * beta and then
// sum into the gradient associated with each label.
// NOTE computes gradient w.r.t UNNORMALIZED final layer activations.
// Assumed passed in grads are already zeroed!
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                            ProbT log_partition,
                                            int L, int T, 
                                            const int* const labels,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output) {
    int start = L-1, end = L;

    std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

    // //set the starting values in the beta column at the very right edge
    for (int i = start; i < end; ++i) {
        betas[i] = std::log(probs[labels[i] + (T - 1) * (alphabet_size_ * minibatch_)]);

        //compute alpha * beta in log space at this position in (S, T) space
        alphas[i + (T - 1) * L] += betas[i];

        //update the gradient associated with this label
        //essentially performing a reduce-by-key in a sequential manner
        output[labels[i]] =
                ctc_helper::log_plus<ProbT>()(alphas[i + (T - 1) * L], output[labels[i]]);
    }

    //update the gradient wrt to each unique label
    for (int i = 0; i < alphabet_size_; ++i) {
        int idx3 = (T - 1) * alphabet_size_ * minibatch_ + i;

        if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
            probs[idx3] == 0.0) {
            grad[idx3] = probs[idx3];
        } else { //output[i] have double probs[idx3] value, one forward and one backward, and so need reduce for one more.
            grad[idx3] = probs[idx3] - std::exp(output[i] -
                                                std::log(probs[idx3]) - log_partition);
        }
    }

    //loop from the second to last column all the way to the left
    for(int t = T - 2; t >= 0; --t) {
        int remain = L - (T - t);
        if(remain >= 0)
            start--;
        if(t < L-1)
            end--;

        int endloop = end == L ? end - 1 : end;
        int idx1 = t * L, idx3 = t * (alphabet_size_ * minibatch_);

        std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

        for(int i = start; i < endloop; ++i) {
            ProbT next_sum = ctc_helper::log_plus<ProbT>()(betas[i], betas[(i+1)]);

            betas[i] = next_sum + std::log(probs[labels[i] + idx3]);

            //compute alpha * beta in log space
            alphas[i + idx1] += betas[i];

            //update the gradient associated with this label
            output[labels[i]] =
                    ctc_helper::log_plus<ProbT>()(alphas[i + idx1], output[labels[i]]);
        }

        if (end == L) {
            betas[(L-1)] = betas[(L-1)] + std::log(probs[labels[L-1] + idx3]);
            alphas[(L-1) + idx1] += betas[(L-1)];
            output[labels[L-1]] =
                    ctc_helper::log_plus<ProbT>()(alphas[L-1 + idx1], output[labels[L-1]]);
        }

        //go over the unique labels and compute the final grad
        // wrt to each one at this time step
        for (int i = 0; i < alphabet_size_; ++i) {
            if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
                probs[idx3] == 0.0) {
                grad[idx3] = probs[idx3];
            } else {
                grad[idx3] = probs[idx3] - std::exp(output[i] -
                                                    std::log(probs[idx3]) - log_partition);
            }
            ++idx3;
        }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglike = ctc_helper::log_plus<ProbT>()(loglike, betas[i]);
    }
    return loglike;
}


template<typename ProbT>
ctcStatus_t
CpuCTC<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths) {
    if (activations == nullptr ||
        grads == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    ProbT* probs = static_cast<ProbT *>(workspace_);

    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * maxT;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);;

    //output
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxL * maxT;

    //betas
    per_minibatch_bytes += sizeof(float) * maxL;

    //labels w/blanks
    per_minibatch_bytes += 1 * sizeof(int) * maxL;

    softmax(activations, probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription

        bool mb_status;

        std::tie(costs[mb], mb_status) =
                cost_and_grad_kernel(grads + mb * alphabet_size_,
                                     probs + mb * alphabet_size_,
                                     flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                                     T, L, mb,
                                     bytes_used + mb * per_minibatch_bytes);
    }

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t CpuCTC<ProbT>::score_forward(const ProbT* const activations,
                                         ProbT* costs,
                                         const int* const flat_labels,
                                         const int* const label_lengths,
                                         const int* const input_lengths) {
    if (activations == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    ProbT* probs = static_cast<ProbT *>(workspace_);

    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * maxT;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);
    // int maxS = 2 * maxL + 1;

    //output
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxL * maxT;

    //betas
    per_minibatch_bytes += sizeof(float) * maxL;

    //labels w/blanks
    per_minibatch_bytes += 1 * sizeof(int) * maxL;

    softmax(activations, probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription

        CpuCTC_metadata ctcm(L, T, mb, alphabet_size_, workspace_,
                             bytes_used + mb * per_minibatch_bytes,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));


        if (L > T)
            costs[mb] = ProbT(0);
        else {
            costs[mb] = -compute_alphas(probs + mb * alphabet_size_, L, T, ctcm.labels_w,
                                        ctcm.alphas);
        }

    }

    return CTC_STATUS_SUCCESS;
}
