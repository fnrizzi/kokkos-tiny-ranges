
#ifndef KOKKOS_RANGES_HPP
#define KOKKOS_RANGES_HPP

#include <Kokkos_Core.hpp>

namespace Kokkos{
namespace details{

template<class RangeT>
struct TakeProxy
{
  using memory_space = typename RangeT::memory_space;
  using reference_type = typename RangeT::reference_type;

  RangeT base_ {};
  std::size_t count_ {};

public:
  TakeProxy() = default;

  explicit TakeProxy(RangeT base, std::size_t count)
    : base_(base)
    , count_(count)
  {}

  KOKKOS_FUNCTION
  std::size_t size() const{ return count_; }

  KOKKOS_FUNCTION
  reference_type operator()(int n) const {
    return base_(n);
  }
};

template<class RangeT>
struct ReverseProxy
{
  using memory_space = typename RangeT::memory_space;
  using reference_type = typename RangeT::reference_type;
  RangeT base_ {};

public:
  ReverseProxy() = default;
  explicit ReverseProxy(RangeT base) : base_(base){}

  KOKKOS_FUNCTION
  std::size_t size() const{ return base_.size(); }

  KOKKOS_FUNCTION
  reference_type operator()(int n) const {
    return base_(base_.size()-1-n);
  }
};

template <class Base, class Shifts, class PredType>
struct MyFunc {
  Base m_base;
  Shifts m_shifts;
  PredType m_pred;

  KOKKOS_FUNCTION
  MyFunc(Base b, Shifts s, PredType pred)
    : m_base(b), m_shifts(s), m_pred(pred){}

  KOKKOS_FUNCTION
  void operator()(const int i, int& update,
                  const bool final_pass) const
  {
    const auto& myval = m_base(i);
    if (final_pass) {
      if (m_pred(myval)) {
        m_shifts[update] = i;
      }
    }

    if (m_pred(myval)) {
      update += 1;
    }
  }
};

template<class RangeT, class Pred>
class NonLazyFilterProxy
{
public:
  using memory_space = typename RangeT::memory_space;
  using reference_type = typename RangeT::reference_type;

private:
  RangeT base_ {};
  Pred pred_ {};
  Kokkos::View<int*, memory_space> shifts_;

public:
  NonLazyFilterProxy() = default;

  explicit NonLazyFilterProxy(RangeT base, Pred pred)
    : base_(base), pred_(pred), shifts_("shifts", base.size())
  {
    int count = 0;
    ::Kokkos::parallel_scan("nonlazyfilter", base_.size(),
                            MyFunc(base_, shifts_, pred), count);
    Kokkos::resize(shifts_, count);
  }

  KOKKOS_FUNCTION
  std::size_t size() const{ return shifts_.extent(0); }

  KOKKOS_FUNCTION
  reference_type operator()(int n) const{
    return base_(shifts_[n]);
  }
};

// closures
template<class Pred>
struct NonLazyFilterRangeAdaptorClosure{
  Pred pred_;
  explicit NonLazyFilterRangeAdaptorClosure(Pred pred) : pred_(pred){}

  template <class RangeT>
  auto operator()(RangeT && r) const{
    return NonLazyFilterProxy(std::forward<RangeT>(r), pred_);
  }
};

struct TakeRangeAdaptorClosure{
  std::size_t count_;
  explicit TakeRangeAdaptorClosure(std::size_t count) : count_(count){}

  template <class RangeT>
  auto operator()(RangeT && r) const{
    return TakeProxy(std::forward<RangeT>(r), count_);
  }
};

struct ReverseRangeAdaptorClosure{
  ReverseRangeAdaptorClosure() = default;

  template <class RangeT>
  auto operator()(RangeT && r) const{
    return ReverseProxy(std::forward<RangeT>(r));
  }
};

// adaptors
struct NonLazyFilterRangeAdaptorFn{
  template<class Pred>
  auto operator () (Pred pred) const{
    return NonLazyFilterRangeAdaptorClosure<Pred>(pred);
  }
};

struct TakeRangeAdaptor{
  auto operator () (std::size_t count){
    return TakeRangeAdaptorClosure(count);
  }
};

struct ReverseRangeAdaptor{
  auto operator() (){
    return ReverseRangeAdaptorClosure();
  }
};
}//namespace details

details::NonLazyFilterRangeAdaptorFn nonlazy_filter;
details::TakeRangeAdaptor take;
details::ReverseRangeAdaptor reverse;

template<class R, class T>
auto operator | (R&& r, T const & a){
  return a(std::forward<R>(r));
}

}
#endif
