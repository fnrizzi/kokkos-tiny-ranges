
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <kokkos_ranges.hpp>

auto create_kokkos_view(int n){
  Kokkos::View<int*> v("v", n);
  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  for (int i=0; i<v_h.extent(0); ++i){
    v_h(i) = i;
  }
  Kokkos::deep_copy(v, v_h);
  return v;
}

template<class T>
void print(T a){
  for (int i=0; i<a.size(); ++i){
    std::cout << a(i) << ' ';
  }
  std::cout << '\n';
}

template<class T>
requires Kokkos::is_view_v<T>
void print(T view)
{
  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
  for (int i=0; i<v_h.extent(0); ++i){
    std::cout << v_h(i) << ' ';
  }
  std::cout << '\n';
}

template<class T, class ResultT>
struct ProxyToViewFunc{
  T in_;
  ResultT res_;

  KOKKOS_FUNCTION
  ProxyToViewFunc(T in, ResultT res)
    : in_(in), res_(res){}

  KOKKOS_FUNCTION void operator()(int i) const{
    res_(i) = in_(i);
  }
};

template<class T>
auto proxy_to_host_view_via_parfor(T p)
{
  Kokkos::View<int*> result("r", p.size());
  Kokkos::parallel_for(p.size(), ProxyToViewFunc(p, result));
  auto r_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  return r_h;
}

template<class ViewT, class VecT>
void assert_equal(ViewT view, VecT const & vec){
  ASSERT_TRUE(view.extent(0) == vec.size());
  for (int i=0; i<vec.size(); ++i){
    ASSERT_TRUE(view(i) == vec[i]);
  }
}

struct IsEven{
  KOKKOS_FUNCTION bool operator()(int v) const{
    return v % 2 == 0;
  };
};

// ------------------------------------------
// tests
// ------------------------------------------

TEST(something, filter)
{
  auto kv = create_kokkos_view(10);
  auto p = kv | Kokkos::nonlazy_filter(IsEven());
  auto p_view = proxy_to_host_view_via_parfor(p);
  print(p_view);
  std::vector<int> gold = {0,2,4,6,8};
  assert_equal(p_view, gold);
}

TEST(something, filter_take)
{
  auto kv = create_kokkos_view(10);
  auto p = kv | Kokkos::nonlazy_filter(IsEven()) | Kokkos::take(2);
  auto p_view = proxy_to_host_view_via_parfor(p);
  print(p_view);
  std::vector<int> gold = {0,2};
  assert_equal(p_view, gold);
}

TEST(something, take_filter)
{
  auto kv = create_kokkos_view(20);
  auto p = kv | Kokkos::take(10) | Kokkos::nonlazy_filter(IsEven());
  auto p_view = proxy_to_host_view_via_parfor(p);
  print(p_view);
  std::vector<int> gold = {0,2,4,6,8};
  assert_equal(p_view, gold);
}

TEST(something, reverse)
{
  auto kv = create_kokkos_view(10);
  auto p = kv | Kokkos::reverse();
  auto p_view = proxy_to_host_view_via_parfor(p);
  print(p_view);
  std::vector<int> gold = {9,8,7,6,5,4,3,2,1,0};
  assert_equal(p_view, gold);
}

TEST(something, filter_reverse)
{
  auto kv = create_kokkos_view(10);
  auto p = kv | Kokkos::nonlazy_filter(IsEven()) | Kokkos::reverse();
  auto p_view = proxy_to_host_view_via_parfor(p);
  print(p_view);
  std::vector<int> gold = {8,6,4,2,0};
  assert_equal(p_view, gold);
}

TEST(something, filter_reverse_take)
{
  auto kv = create_kokkos_view(10);
  auto p = kv | Kokkos::nonlazy_filter(IsEven()) | Kokkos::reverse() | Kokkos::take(2);
  auto p_view = proxy_to_host_view_via_parfor(p);
  print(p_view);
  std::vector<int> gold = {8,6};
  assert_equal(p_view, gold);
}
