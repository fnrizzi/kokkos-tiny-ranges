# kokkos-tiny-ranges

Example:

```cpp
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven());
Kokkos::parallel_for(p.size(), /*functor*/);
```

```cpp
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven()) | Kokkos::reverse() | Kokkos::take(2);
Kokkos::parallel_for(p.size(), /*functor*/);
```