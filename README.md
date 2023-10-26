# kokkos-tiny-ranges

Example:

```cpp

struct IsEven{
  KOKKOS_FUNCTION bool operator()(int v) const{
    return v % 2 == 0;
  };
};

template<class RangeT>
struct MyFunc{
  RangeT r_;

  MyFunc(RangeT r) : r_(r){}

  KOKKOS_FUNCTION void operator()(int i) const{
   auto values = r_(i);
   // do soemthing
  }
};

// 1.
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven());
Kokkos::parallel_for(p.size(), MyFunc(p));

// 2.
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven()) | Kokkos::reverse() | Kokkos::take(2);
Kokkos::parallel_for(p.size(), MyFunc(p));
```
