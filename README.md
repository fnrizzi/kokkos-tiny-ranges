# kokkos-tiny-ranges

## Disclaimer 

This is very much WIP, purely just a prototype to play with (missing pretty much all the fine-grained details, proper semantics, etc...)
but it works! 

## Snippets 

```cpp

struct IsEven{
  KOKKOS_FUNCTION bool operator()(int v) const{ return v % 2 == 0; };
};

template<class RangeT>
struct MyFunc{
  RangeT r_;

  KOKKOS_FUNCTION
  MyFunc(RangeT r) : r_(r){}

  KOKKOS_FUNCTION void operator()(int i) const{
   auto value = r_(i);
   // do soemthing
  }
};

// example 1
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven());
Kokkos::parallel_for(p.size(), MyFunc(p));

// example 2
Kokkos::View<int*> view("v", 110);
auto p = view | Kokkos::nonlazy_filter(IsEven()) | Kokkos::reverse() | Kokkos::take(2);
Kokkos::parallel_for(p.size(), MyFunc(p));
```
