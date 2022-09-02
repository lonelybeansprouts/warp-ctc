#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <stdlib.h>
#include <torch/extension.h>

using Tensor = torch::Tensor;
using ScalarType = torch::ScalarType;
using IntArrayRef = torch::IntArrayRef;


class ThreadPool {
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}



std::shared_ptr<ThreadPool> get_thread_pool(){
  static std::shared_ptr<ThreadPool> thread_pool = NULL;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (thread_pool != NULL) return thread_pool;
  char* thread_num_str = getenv("BFCTC_THREAD_NUM");
  int thread_num;
  if (thread_num_str){
    std::string __(thread_num_str);
    std::stringstream ss(__);
    ss >> thread_num;
  }
  else{
    thread_num = 8; //default
  }

  std::cout<<"bfctc thread number is: "<<thread_num<<"\n";
  thread_pool = std::make_shared<ThreadPool>(thread_num);
  return thread_pool;
}


// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1]) note that no bound-checking is done
template<typename target_t>
static inline int64_t get_target_prime(target_t* target, int64_t offset, int64_t stride, int64_t idx) {
    return target[offset + stride * idx];
}

// This kernel is a relatively straightforward implementation of the alpha calculation in the forward backward algorithm (section 4.1).
// A (minor) twist is that we are using log-calculations to enhance numerical stability (log_probs and log_alpha).
// The function returns the loss and the alphas, the alphas are kept for the backward step. The wrapper (ctc_loss below) hides
// the alphas from the user by only returning the loss.

// log_probs: input_len x batch_size x num_labels
// targets [int64]: batch_size x target_length OR sum(target_lengths)
template<typename scalar_t, typename target_t>
std::tuple<Tensor, Tensor> ctc_loss_cpu_template(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths) {
  auto thread_pool = get_thread_pool();
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  int64_t batch_size = log_probs.size(1);

  size_t tg_target_stride;
  int64_t max_target_length = 0;
  std::vector<int64_t> tg_batch_offsets(batch_size);
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
         max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    // checkSize(c, targets_arg, 0, pos);
  }else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
  }


  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), max_target_length}, log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  auto lpp  = log_probs.permute({1,0,2});
  auto log_probs_a_global = lpp.accessor<scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<scalar_t, 3>();
  auto targets_data = targets.data_ptr<target_t>();
  auto neg_log_likelihood_a = neg_log_likelihood.accessor<scalar_t, 1>();

  // alpha calculation for the first row, the three equations for alpha_1 above eq (6)
  // first the default
  log_alpha.narrow(1, 0, 1).fill_(neginf);


  auto process = [&](int64_t b)->long{
        int64_t input_length = input_lengths[b];
        int64_t target_length = target_lengths[b];
        auto log_probs_a = log_probs_a_global[b];
        auto log_alpha_a = log_alpha_a_global[b];
        int64_t tg_batch_offset = tg_batch_offsets[b];

        if (input_length<target_length){
          neg_log_likelihood_a[b] = std::numeric_limits<scalar_t>::infinity();
          std::cerr<<"sample idx:"<<b<<" input_length<target_length, ignored !"<<"\n";
          return b;
        }

        // the first two items of alpha_t above eq (6)
        log_alpha_a[0][0] = log_probs_a[0][get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 0)];

        // now the loop over the inputs
        for (int64_t t=1; t<input_length; t++) {
            for (int64_t s=0; s<target_length; s++) {
            auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s);
            // this loop over s could be parallel/vectorized, too, but the required items are one index apart
            // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
            // for the cuda implementation, that gave a speed boost.
            // This is eq (6) and (7), la1,2,3 are the three summands. We keep track of the maximum for the logsumexp calculation.

            scalar_t la1 = log_alpha_a[t-1][s];
            scalar_t lamax = la1;
            scalar_t la2;
            if (s > 0) {
                la2 = log_alpha_a[t-1][s-1];
                if (la2 > lamax)
                    lamax = la2;
            } else {
                la2 = neginf;
            }

            if (lamax == neginf) // cannot do neginf-neginf
                lamax = 0;
            // this is the assignment of eq (6)
            log_alpha_a[t][s] = std::log(std::exp(la1-lamax)+std::exp(la2-lamax))+lamax + log_probs_a[t][current_target_prime];
            }
        }

        neg_log_likelihood_a[b] = -log_alpha_a[input_length-1][target_length-1];
        return b;
  };

  std::queue<std::future<long>> results;
  for (int64_t batch_idx=0; batch_idx<batch_size; batch_idx++){
    results.push(thread_pool->enqueue(process, batch_idx));
  }
  while (results.size() > 0) {//sychronize
    results.front().get();
    results.pop();
  }

  return std::make_tuple(neg_log_likelihood, log_alpha);
}


// // This is the backward. It consists of two phases:
// // a) computing the beta analogous to the alphas in the forward (backward half of the forward-backward algorithm) (eq (10) and (11))
// // b) collecting the per-activation characters for all s and wrapping the gradient (eq (16), the collection is the sum)
template<typename scalar_t, typename target_t>
std::tuple<Tensor, Tensor>  ctc_loss_backward_cpu_template(const Tensor& grad_out, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                                      const Tensor& neg_log_likelihood, const Tensor& log_alpha, bool zero_infinity=true) {
  auto thread_pool = get_thread_pool();
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  int64_t max_input_length = log_probs.size(0);
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  Tensor grad = at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // at this point, this is log of empty sum

  // The admin bits. We don't do much checking and assume that the forward did.
  int64_t tg_target_stride;
  int64_t max_target_length;
  std::vector<int64_t> tg_batch_offsets(batch_size);

  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length = targets.size(1);
  }

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // could be optimized to use only 2 rows
  auto lpp  = log_probs.permute({1,0,2});
  auto log_probs_a_global = lpp.accessor<scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<scalar_t, 3>();
  auto log_beta_a_global = log_beta.accessor<scalar_t, 3>();
  auto gp = grad.permute({1,0,2});
  auto grad_a_global = gp.accessor<scalar_t, 3>();
  auto targets_data = targets.data_ptr<target_t>();

  auto process = [&](int64_t b)->long{
    scalar_t nll = neg_log_likelihood.accessor<scalar_t, 1>()[b];
    if (zero_infinity &&  nll == std::numeric_limits<scalar_t>::infinity()) {
        grad.narrow(1, b, 1).zero_();
        return b;
    }

    auto log_probs_a = log_probs_a_global[b];
    auto log_alpha_a = log_alpha_a_global[b];
    auto log_beta_a = log_beta_a_global[b];
    auto grad_a = grad_a_global[b];
    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    int64_t tg_batch_offset = tg_batch_offsets[b];

    // the initialization of beta before eq (10)
    // here we do the fill for each batch item separately, as the input lengths will differ, so the t in which
    // we start varies
    log_beta.narrow(0, b, 1).narrow(1, input_length-1, 1).fill_(neginf);
    auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, target_length-1);
    log_beta_a[input_length-1][target_length-1] = log_probs_a[input_length-1][current_target_prime];
    grad_a[input_length-1][current_target_prime] = log_alpha_a[input_length-1][target_length-1] + log_beta_a[input_length-1][target_length-1];


    // now loop applying eq (10) / (11)
    for (int64_t t=input_length-2; t>=0; t--) {
      // this loop over s could be parallel/vectorized and doesn't really need to be descending...
      // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
      // for the cuda implementation, that gave a speed boost.
      for (int64_t s=target_length-1; s>=0; s--) {
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s);
          scalar_t lb1 = log_beta_a[t+1][s];
          scalar_t lbmax = lb1;
          scalar_t lb2;
          if (s < target_length-1) {
            lb2 = log_beta_a[t+1][s+1];
            if (lb2 > lbmax)
                lbmax = lb2;
          } else {
            lb2 = neginf;
          }

          if (lbmax == neginf)
            lbmax = 0;

          log_beta_a[t][s] = std::log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax))+lbmax + log_probs_a[t][current_target_prime];
          // one might check whether one can vectorize this better when done after the t-loop...
          // now that we have beta, we fill in the sum of alpha*beta in eq (16)
          // in contrast to the cuda implementation, we only parallelize over the batch, so we don't have a concurrency
          // issue (several s can map to the same target character)
          // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
          scalar_t log_alpha_beta =  log_alpha_a[t][s] + log_beta_a[t][s];
          scalar_t &lcab = grad_a[t][current_target_prime];
          if (lcab == neginf) {
            lcab = log_alpha_beta;
          } else {
            scalar_t max = std::max(lcab, log_alpha_beta);
            lcab = std::log(std::exp(lcab-max)+std::exp(log_alpha_beta-max))+max;
          }
      }
    }

    // now grad has the sum of eq (16)
    // now we wrap up the calculation by adding in the remaining items of eq (16)
    // this could be a great target for further vectorization.
    // grad is the output gradient, nll is the loss. Note that the likelihood -nll is the Z of eq (16)
    scalar_t gr =  grad_out.accessor<scalar_t, 1>()[b];
    for (int64_t t = 0; t < input_length; t++) { // or go for the full thing?
      for (int64_t c = 0; c < num_labels; c++) {
        scalar_t& res = grad_a[t][c];
        scalar_t lp = log_probs_a[t][c];
        res = (std::exp(lp)-std::exp(res + nll - lp)) * gr;
      }
    }
    // zero the remainder
    if (input_length < max_input_length) {
      grad.narrow(0, input_length, max_input_length - input_length).narrow(1, b, 1).zero_();
    }
    return b;
  };
  
  std::queue<std::future<long>> results;
  for (int64_t batch_idx=0; batch_idx<batch_size; batch_idx++){
    results.push(thread_pool->enqueue(process, batch_idx));
  }
  while (results.size() > 0) {//sychronize
    results.front().get();
    results.pop();
  }

  return std::make_tuple(grad, log_beta);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf_ctc_forward", &ctc_loss_cpu_template<float, int64_t>, "BF CTC Loss function forward with cpu");
  m.def("bf_ctc_backward", &ctc_loss_backward_cpu_template<float, int64_t>, "BF CTC Loss function backward with cpu");
}


