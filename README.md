# kokkos-tiny-ranges

Example:

```cpp

struct IsEven{
  KOKKOS_FUNCTION bool operator()(int v) const{
    return v % 2 == 0;
  };
};

Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven());
Kokkos::parallel_for(p.size(), /*functor*/);

// chain more
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven()) | Kokkos::reverse() | Kokkos::take(2);
Kokkos::parallel_for(p.size(), /*functor*/);
```