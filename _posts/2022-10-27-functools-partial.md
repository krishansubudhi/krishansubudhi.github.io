---
comments: true
author: krishan
layout: post
categories: python
title: functools partial
description: Explains how functools partial works with simple examples. 
---

The `functools` module is for higher-order functions: functions that act on or return other functions. In general, any callable object can be treated as a function for the purposes of this module.

[Official python documentation](https://docs.python.org/3/library/functools.html#functools.partial)

`functools.partial` Return a new partial object which when called will behave like func called with the positional arguments args and keyword arguments keywords.
- If more arguments are supplied to the call, they are **appended** to args.
- If additional keyword arguments are supplied, they **extend and override keywords**.

```python
import functools
print(functools.partial.__doc__)
```

    partial(func, *args, **keywords) - new function with partial application
        of the given arguments and keywords.
    





```python
def foo(name):
  print(f'Inside foo : name = {name}')
foo('Krishan')
```

    Inside foo : name = Krishan


## Partial with default positional argument


```python
foo_with_default = functools.partial(foo, 'Krishan')
```


```python
foo_with_default()
```

    Inside foo : name = Krishan

If `name` is passed again as keyword argument or positional arg, the decorated function will throw error.

```python
foo_with_default(name = 'Ram')
```


    ---------------------------------------------------------------------------

    TypeError: foo() got multiple values for argument 'name'



```python
foo_with_default('Ram')
```


    ---------------------------------------------------------------------------

    TypeError: foo() takes 1 positional argument but 2 were given


## Partial with default Keyword argument


```python
foo_with_default = functools.partial(foo, name = 'Krishan')
```

If `name` is passed again as a keyword argument, the kwargs gets updated before calling `foo` (the function being decorated.).

```python
foo_with_default(name = 'Ram')
```

    Inside foo : name = Ram

Same is not true for passing the argument as positional argument as positional arguments get appended which results in error.

```python
foo_with_default('Ram')
```


    ---------------------------------------------------------------------------

    TypeError: foo() takes 1 positional argument but 2 were given

