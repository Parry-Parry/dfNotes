$$\newcommand{\vec}[1]{{\bf #1} } 
\newenvironment{examinable}{}{{\ \LARGE[\spadesuit]}}
\newcommand{\real}{\mathbb{R}}
\newcommand{\expect}[1]{\mathbb{E}[#1]}
\DeclareMathOperator*{\argmin}{arg\,min}
$$
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>""")

# Database Fundamentals 2020
## Week 1

### ndarrays

* The fundamental data type for this course is the **multidimensional numerical array** (of floating point numbers). 
* This is a very powerful data type, although simple in structure, there are great many operations that can be done elegantly with an array.
* We will call these arrays **ndarrays** (for *n-dimensional arrays*) or sometimes **tensors** (in reference to the mathematical object which generalises vectors and matrices to higher orders), which some people use. 

* Numerical arrays sound boring. But they are arguably the most "fun" data structure. Images, sounds, videos are all most easily worked with as arrays of numbers. 
	* An *image* is a 2D array of brightness values;
	* a *sound* is a 1D array of sound pressure levels; 
	* a *video* is a 3D array of brightness values (x, y, and time).

#### Scientific data

* Scientific data \(e.g. from physics experiments, weather models, even models of how people choose search terms on Google\) can often be most conveniently represented as numerical arrays. 
* The kind of operations we want to do to scientific data (e.g. find the weather most similar today in the historical record) are easily expressed as array operations.

#### 3D graphics

* 3D computer graphics, as you would encounter in a game or VR, usually involves manipulating **geometry**. 
* Geometry is typically specified as simple geometric shapes, like triangles. 
* These shapes are made up of points -- **vertices** -- typically with an \[x,y,z\] location. 
* Operations like moving, rotating, scaling of objects are operations on big arrays of these vertices.

* By representing data as a numerical array, we can extend the operations we apply to single numbers \(like integers or floating points\) to entire arrays of numbers

#### Mathematical power

* There is a rich set of mathematical abstractions that work on spaces defined over array-valued elements. 
* For example, **linear algebra** provides tools to work with 1D arrays \(*vectors*\) and 2D arrays \(*matrices*\) and can be used to solve many difficult problems. 
* Having these types represented as basic types in a programming language makes working with linear algebraic problems vastly easier.

#### Efficiency

* Numerical arrays are both:
	* **compact** \(they store data in a very memory efficient way\) 
	* **computationally efficient** \(it is possible to write code that manipulates arrays extremely quickly\)

#### Deep learning

* The key to deep learning is to be able to represent data as arrays of numbers and to do **all** computations as array operations. 
* That is we perform operations that act on all elements of an array simultaneously.

##### Vectorisation: one operation, many data

* The practice of writing code which acts on arrays of values simultaneously is called **vectorised computation**. 
* It is a special case of **parallel** computing, where we restrict ourselves to numerical operations on fixed size arrays. 
* Modern CPUs have numerous **vectorised** instructions to perform the same operation on many numbers at once \(e.g. MMX, SSE, SSE2, SSE3 on x86, NEON on ARM, etc.\). 
	* This is called **Single Instruction Multiple Data**.

##### GPUs are array processors

* **But** they are effectively big groups of very simple processors, which are able to deal very well with data in numerical arrays, but are very slow when working with other data structures. 
* Anything that can be written as an operation on numerical arrays can be done at lightning speed on a GPU. 
* In fact, GPUs are basically devices that can do computations on numerical arrays, **and that's it**. 
* To write \(efficient\) GPU code, you need to write code in terms of numerical arrays.

#### Spreadsheet\-like computation

* Array types are much like entire *spreadsheets in a single variable*, which you can perform standard spreadsheet operations on, like: 
	* tallying up columns 
	* selecting values which have a certain range
	* plotting charts
	* joining together several sheets  
* The abstraction of array types makes it easy to do what are complex operations with a standard spreadsheet. 
* And they work on data beyond just 2D tables.

### Typing and shapes of arrays

#### Vector, matrix, tensor

* **ndarrays** can have different *dimensions*; sometimes called *ranks*, as in a "*rank\-3 tensor*" or "3D array" , meaning an array with rows, columns and channels.
* We call a 1D array of values a **vector**
* A 2D array of values is called a **matrix**, and is formed of rows and columns
* Any array with more than 2 dimensions is just called an **nD array**  \(**n d**imensional array\) or sometimes a **tensor**. 
	* There isn't a convenient mathematical notation for tensors.

_It is often easiest to think of tensors as arrays of matrices or vectors \(e.g a 3D tensor is really a stack of 2D matrices, a 4D arrays is a grid of 2D matrices, a 5D array is a stack of those grids, etc.\)_

* We typically don't encounter tensors with more than 6 dimensions, as these would require enormous amounts of memory to store and don't correspond to many real\-world use cases. 

#### Axes

* We often refer to specific dimensions as **axes** or **dimensions**. 
* For example a matrix \(a 2D array\) has two axes: **rows** \(axis 0\) and **columns** \(axis 1\). 
* A vector has just one axis, axis 0. 
* A 4D tensor has 4 axes, which are indexed 0, 1, 2, 3.
* Many operations we can do can be selectively applied only on certain axes, which is a very useful way to specify the effect of an operation.

### Array operations

* Unlike most of the data structures you are familiar with \(lists, dictionaries, strings\), there are **many** operations defined on arrays. 
* Some of these are convenience operations, but there are many fundamental operations as well. 
	* **Slice**: slice out rectangular regions, for reading or writing.
    	* Chop out second to tenth rows of a 2D matrix: `x[1:9, :]`
    	* Set the second column to 0 `x[:,1] = 0`
	* **Filter**: find values matching criteria.
    	* select elements of x where x is negative `x[x<0]`
	* **Reduce**: aggregate across dimensions.
    	* compute sum of each column;  `np.sum(x, axis=0)`
	* **Map**: apply functions or arithmetic operations elementwise
    	* add 1 to every element of x `x+1` 
    	* add x and y `x+y` 
    	* take the sine of every element of x `np.sin(x)`
	* **Concatenate and repeat**:
    	* stick x and y together one on top of the other; np.concatenate\(\[x,y\], axis=0\)
    	* repeat x 8 times across the columns; np.tile\(x, \[1,8\]\)
	* **Generate**:
    	* create arrays of all zeros: `np.full((8,8),0)`
    	* create "counting" arrays: `np.arange(10)`
    	* load arrays from files or save them to disk.
	* **Reorder**
    	* reverse/flip axes
    	* sort axes
    	* exchange rows/columns \(transpose\)

#### No explicit iteration

* We operate on arrays without writing explicit iterations over their elements wherever possible. 
* This has two effects:
	* The code is **much simpler**
	* * The code is **much faster**. The operations can be run in accelerated routines, or with hardware acceleration, e.g. on the GPU.

### Vectors and matrices

As well as being convenient to implement in silicon, arrays correspond to rich mathematical objects.

#### Geometry of vectors

* 1D arrays can be used to represent vectors, and vectors have a mathematical structure. 
* This structure of vectors is essential in modeling physical systems, building information retrieval systems, machine learning and 3D rendering. 
* Vectors have length, direction; they can be added, subtracted, scaled; various products are defined on vectors. 
* Arrays are often used to represent vectors; for example a 2D array might be used to store a sequence of vectors \(perhaps positions in space\), which could be operated on simultaneously.

* We write a vector a bold letter lower case symbol: $\bf x$ \(other notations include an arrow or bar above an upper case symbol\).

#### Signal arrays

* 1D arrays can also be used to represent **signals**; that is sequences of measurements over time. 
* Signals include images, sounds, and other time series. Signals can be scaled, mixed, chopped up and rearranged, filtered, and processed in many other ways.

* The use of arrays to represent sequences and vectors is not exclusive; some operations use both representations simultaneously. 
* We typically write a signal as x\[t\].


#### Algebra of matrices

* 2D arrays are matrices, which have an algebra: **linear algebra**. 
* A matrix represents a **linear map**, a particular kind of function which operates on vectors \(in **vector space**\), and the operation of the function is completely defined by the elements of that matrix. 

* Linear algebra is extremely rich can is used to perform many essential computations. 
* For example, almost all of 3D rendering involves matrix operations \(e.g. coordinate transformation\). 
* It has many interesting features; multiplication is defined such that it **applies** the map when multiplying with vectors and **composes** the map when multiplying with other matrices. 

We write a matrix as an upper case symbol A. 
    
#### Mathematical operations

We consequently have specialised mathematical operations we can apply to arrays

* **Vector operations**: operations which apply geometric effects to vectors: dot product, cross product, norm
    * for example, getting the Euclidean length of a vector
* **Matrix operations**: linear algebra operations like multiplication, transpose, inverse, matrix exponentials, decompositions
    * for example, multiplying together two 3D transformation matrices
* **Signal processing operations**: signal processing relevant operations like convolution, Fourier transform, numerical gradients, cumulative summation
    * for example, blurring an image using a convolution

### Statically typed, rectangular arrays: **ndarrays**

**ndarrays** \(n\-dimensional arrays\) represent sequences. However, arrays are not like lists. 

They have

* **fixed, predefined** size \(or "shape"\)
* **fixed, predefined** type \(all elements have the same type\)
* they can only hold numbers \(typically integers or floating\-point\)
* they are inherently multidimensional
* they are required to be "rectangular" -- a 2D array must have the same number of columns in each row

* The type of an array has to be specified very precisely; for example, we have to specify the precision of floating point numbers if we use floats \(usually the options are 32 or 64 bits\).
* Arrays cannot \(usually\) be extended or resized after they have been created. 
* To have the effect of changing the size of an array, we must create a new array with the right size and copy in \(a portion of\) the old elements.
* Arrays **are** typically mutable though, and the values they hold can be changed after creation. So we can write new values into an existing array.

#### Reasons for typing

* **ndrrays** are a fairly  thin wrapper around raw blocks of memory. This is what makes them compact and efficient
* Because of this restriction on typing, numerical arrays are **much** more efficiently packed into memory, and operations on them can be performed **extremely** quickly \(several orders of magnitude faster than plain Python\).
* The same is true for other platforms and languages: numerical arrays are normally implemented to be the fastest and smallest possible structure for representing blocks of numbers.

## Practical array manipulation

### Shape and dtype

* Every array is characterised by two things:
	* the type of its elements: the **dtype** \(e.g.` float64`\)
	* its **shape**: that is, its dimensions. For example, 32x8

#### Order

We always discuss the shape of arrays in the order  
* rows
* columns
* depth/frames/channels/planes/...

**This ordering is important: remember it!**

#### Indexing and axes  

* We index from **0**, so element \[0,1\] of an array means *first row, second column*
* The **dimensions** of an array are often called its **axes**. 
	* Do not confuse this with axes or dimension of a vector space! The number of dimensions of an array is sometimes called its **rank**

* A scalar is a rank 0 tensor
* A vector is a rank 1  tensor \(1 dimensional array\)
* A matrix is a rank 2 tensor \(2 dimensional array\)
* A stack of matrices is a rank 3 tensor \(3 dimensional array\)
* and so on...

### Dtypes

* **dtype** just stands for the *d*ata *type*, and it is the data type of every element in an array \(what kind of number it is\). 
* Every element has the same *dtype*; it applies to the whole array.

Common *dtypes* are:  
* `float64` double\-precision float numbers
* `float32` single\-precision float numbers
* `int32` signed 32 bit integers
* `uint8` unsigned 8 bit integers

### Tabular Data

* Many other data can be naturally thought of as arrays. 
* For example, a very common structure is a spreadsheet like arrangement of data in tables, with rows and columns.
* Each row is an **observation**, and each column is a **variable**

### Creating arrays
#### Converting and copying: np.array

New arrays can be created in several ways:

* Converted from another sequence type, like a list: `np.array()` does this.
* Created blank and filled with some value.  This is often essential in creating temporary variables, for example to accumulate results into.
* Filled with random values.
* Loaded from disk.

`np.array()` takes a sequence and converts it into an array; this can be, for example, a list.
It works for multidimensional arrays as well, given nested sequences. 

### Mutability and copying

`np.array()` can take any sequence, including another ndarray. So it can be used to copy arrays:

This is important, because NumPy arrays are **mutable**, and if several variables refer to the **same** array, the effects might not be what you expect

_We need to explicitly copy arrays if we want to work on a new array_


### Blank arrays

There are a number of methods all of which do essentially the same thing; allocate a new array with a given shape, and possibly fill it with a value.

* `np.empty(shape)`, which allocates memory for an array, but does not initialise it
* `np.zeros(shape)`,  which initialises all elements to 0
* `np.ones(shape)`,  which initialises all elements to 1
* `np.full(shape, value)` which initialises all elements to `value`

These are just calling `np.empty(shape)` to create a new array and then filling it with a given value.

#### Blank like

Similarly, we can create blank arrays with the same shape and dtype as an existing array using the `_like` variants. `y = np.zeros_like(x)` 

### Random arrays

We can also generate random numbers to fill arrays. Many algorithms use arrays of
random numbers as their basic "fuel". 

* `np.random.randint(a,b,shape)` creates an array with uniform random *integers* between a and \(excluding\) b
* `np.random.uniform(a,b,shape)` creates an array with uniform random *floating point* numbers between a and b
* `np.random.normal(mean,std,shape)` creates an array with normally distributed random floating point numbers between with the given mean and standard deviation.

### arange

We can create a vector of increasing values using `arange` \(**a**rray **range**\), which works like the built in Python function `range` does, but returns an 1D array \(a vector\) instead of a list.

`np.arange()` takes one to three parameters:
* `np.arange(end)`  -- returns a vector of numbers 0..end-1
* `np.arange(start, end)`  -- returns a vector of numbers start..end-1
*  `np.arange(start, end, step)` --returns a vector of numbers start..end-1, incrementing by step \(which may be **negative** and/or **fractional**!\)

### Linspace

* `np.arange` is useful for generating evenly spaced values, but it is parameterised in a form that can be awkward.
* `np.linspace(start, stop, steps)` is a much easier to use alternative. `linspace` stands for "**lin**early **space**d", and it generates `steps` values between `start` and `stop` **inclusive**. 

### Loading and saving arrays

I/O with arrays is a critical part of any numerical computation system. There are a huge number of ways to store and recall arrays, including:  
* simple text formats like CSV \(comma separated values\)
* binary formats for storing single or multiple arrays like mat and npz
* specialised scientific data formats like HDF5 \(often used for huge datasets\)
* domain-specific formats like images \(png, jpg, etc.\), sounds \(wav, mp3, etc.\), 3D geometry \(obj, ...\)

#### Text files  
*  `np.loadtxt(fname)` and 
* `np.savetxt(arr, fname)` 

work on simple text files.

### Slicing and indexing arrays

Arrays can be indexed like lists or sequences \(in Python, this uses square brackets \[\]\), but arrays can have **multidimensional** indices. These are indices which are really tuples of values.

This means we write the variable, with the index in square brackets, where the index might have comma separated values. Indices start at **zero**!

Indexing, and its counterpart slicing, are two of the most important array operations.

The general format follows the same principles as `arange()`, taking 0\-3 parameters separated by a `:`

    start : stop : step
    
Where `start` is the index to start from, `stop` is the end, and `step` is the jump to make between each step. **Any of these parts can be omitted**.

* If there is no colon, this specifies a specific index, for example x\[0\] or x\[18\]
* If there is one colon, this is a range; for example x\[2:5\] or x\[:4\] or x\[0:\]
* If there are two colons, this is a range with a step, like x\[0:10:2\], or x\[:10:2\] or x\[::-1\]

* If `start` is missing, it defaults to 0
* If `end` is missing, it defaults to the last element
* If `step` is missing, it defaults to 1. You don't need to include the second colon if you are omitting step, though it's not an error to do so.

#### Negative indices

Negative indices mean *counting from the end*; so `x[-1]` is the last element, `x[-2]` is the second last, etc.

If we specify an axis as an index \(no range\), we get back a  **slice** of that array, with fewer dimensions. If `x` is 2D, then `x[0,:]` is the first row \(a 1D vector\), and `x[:,0]` is the first column \(a 1D vector\).

### Slicing versus indexing

* **Slicing** does not change the rank of an array. It selects a rectangular subset with the same number of dimensions.
* **Indexing** reduces the rank of an array \(usually\). It selects a rectangular subset where one dimension is a singleton, and removes that dimension.

### Boolean tests

We can do any test \(like equality, greater than, etc.\) on arrays as well: the result is a **Boolean array**, with the same shape as the original array \(despite how they appear, these are actually numeric arrays internally\)

### Rearranging arrays

Arrays can be transformed and reshaped; this means that they keep the same elements, but the arrangements of the elements are changed. For example, the sequence 
    
    [1,2,3,4,5,6]
    
could be rearranged into

    [6,5,4,3,2,1]

which has the same elements but now ordered backwards. 

These operations are often very useful to rearrange arrays so that broadcasting operations can be carried out effectively.

### Transposition

A particularly useful transformation of an array is the **transpose** which exchanges rows and columns \(this isn't the same as rotating 90 degrees!\). There is special syntax for this because it is so often used:

We write `x.T` to get the transpose of `x`.

Transposition has *no effect* on a 1D array, and it reverses the order of all dimensions in >2D arrays.

Note that transposing is a very fast operation -- it does not \(normally\) copy the array data, but just changes how it is accessed, and thus has virtually no time penalty, and completes in O\(1\) time. 

### Flip, rotate

As well as transposition, arrays can also be flipped and rotated in a single operation using indexing \(there are also convenience functions like `fliplr` and `rot90` but we will keep it simple here\)
    
### Cut+tape operations
#### Joining and stacking

We can also join arrays together. But unlike simple structures like lists, we have to explicitly state on which **dimension** we are going to join. And we must adhere to the rule that the output array has rectangular shape; we can't end up with a "ragged" array. \(arrays are *always* rectangular\)

#### concatenate and stack

Because arrays can be joined together along different axes, there are two distinct kinds of joining:
* We can use `concatenate` to join along an *existing* dimension;
* or `stack` to stack up arrays along a *new dimension.*

### 2D matrix stacking shorthand

As a shorthand, there are three defined stacking operations for specific axes when working with 2D matrices:
* `np.hstack()` stacks horizontally
* `np.vstack()` stacks vertically
* `np.dstack()` stacks "depthwise" \(i.e. one matrix on top of another\)

All of these operate on 2D matrices

### Tiling

We often need to be able to **repeat** arrays. This is called **tiling** and `np.tile(a, tiles)` will repeat `a` in the shape given by `tiles`, joining the result together into a single array.

### Selection and masking

Comparisons between arrays result in Boolean arrays

These Boolean arrays have many useful applications in **selecting** specific data, or alternatively **masking** specific data. Selection and masking are basic operations.

#### where, nonzero

We can use a Boolean array to select elements of an array with `np.where(bool, a,b)` which selects `a` where `bool` is True  and `b` where `bool` is False. `bool`, `a` and `b` must be the same shape, or be broadcastable to the right shape.

##### nonzero

We can convert a boolean array to an array of **indices** with `nonzero`

### Fancy indexing

This brings us to the next operation -- an extension of indexing, which allows us to index arrays with arrays. It is a very powerful operator, because we can select *irregular* parts of an array and perform operations on them.

#### Index arrays

For example, an array of **integer** indices can be used as an index directly

### Boolean indexing
As well as using indices, we can directly index arrays with boolean arrays.


For example, if we have an array 

    x = [1,2,3]
    
and an array
   
    bool = [True, False, True]
   
then `x[bool]` is the array:

    [1,3]
    
Note that this is pulling out *irregular* parts of the array \(although the result is always guaranteed to be a rectangular array\).


### Map

This is a special case of a **map**: the application of a function to each element of a sequence.

    
There are certain rules which dictate what operations can be applied together.

* For single argument operations, there is no problem; the operation is applied to each element of the array
* If there are more than two arguments, like in `x + y`, then `x` and `y` must have **compatible shapes**. This means it must be possible to pair each element of `x` with a corresponding element of `y`


#### Same shape

In the simplest case, `x` and `y` have the same shape; then the operation is applied to each pair of elements from `x` and `y` in sequence.

#### Not the same shape

If `x` and `y` aren't the same shape, it might seem like they cannot be added \(or divided, or "maximumed"\). However, NumPy provides **broadcasting rules** to allow arrays to be automatically expanded to allow operations between certain shapes of arrays.

#### Repeat until they match

The rule is simple; if the arrays don't match in size, but one array can be *tiled* to be the same size as the other, this tiling is done implicitly as the operation occurs. For example, adding a scalar to an array implicitly *tiles* the scalar to the size of the array, then adds the two arrays together \(this is done much more efficiently internally than explicitly generating the array\).

The easiest broadcasting rule is scalar arithmetic: `x+1` is valid for any array `x`, because NumPy **broadcasts** the 1 to make it the same shape as `x` and then adds them together, so that every element of `x` is paired with a 1.

Broadcasting always works for any scalar and any array, because a scalar can be repeated however many times necessary to make the operation work.

You can imagine that `x+1` is really `x + np.tile(1, x.shape)` which works the same, but is much less efficient

### Broadcasting

So far we have seen:
* **elementwise array arithmetic** \(both sides of an operator have exactly the same shape\) and 
* **scalar arithmetic** \(one side of the operator is a scalar, and the other is an array\). 

This is part of a general pattern, which lets us very compactly write operations between arrays of different sizes, under some specific restrictions. 

**Broadcasting** is the way in which arithmetic operations are done on arrays when the operands are of different shapes.

1. If the operands have the same number of dimensions, then they **must** have the same shape; operations are done elementwise. `y = x + x`
1. If one operand is an array with fewer dimensions than the other, then if the *last dimensions* of the first array match the shape as the second array, operations are well-defined. If we have a LHS of size \(...,j,k,l\) and a RHS of \(l\) or \(k,l\) or \(j,k,l\) etc., then everything is OK.

This says for example that:

    shape (2,2) * shape(2,) -> valid
    shape (2,3,4) * shape(3,4) -> valid
    shape (2,3,4) * shape(4,) -> valid
    
    shape (2,3,4) * shape (2,4) -> invalid 
    shape (2,3,4) * shape(2) --> invalid
    shape (2,3,4) * shape(8) --> invalid

#### Broadcasting is just automatic tiling

When broadcasting, the array is *repeated* or tiling as needed to expand to the correct size, then the operation is applied. So adding a \(2,3\) array and a \(3,\) array means repeating the \(3,\) array into 2 identical rows, then adding to the \(2,3\) array.


### Reduction

**Reduction** \(sometimes called **fold**\) is the process of applying an operator or function with two arguments repeatedly to
some sequence. 

For example, if we reduce [1,2,3,4] with `+`, the result is `1+2+3+4 = 10`. If we reduce `[1,2,3,4]` with `*`, the result is `1*2*3*4 = 24`. 

**Reduction: stick an operator in between elements**

    1 2 3 4
    5 6 7 8
   
Reduce on columns with "+":

    1 + 2 + 3 + 4  =  10
    5 + 6 + 7 + 8  =  26
    
Reduce on rows with "+":

    1 2 3 4
    + + + +
    5 6 7 8
    
    = 
    6 8 10 12
    
Reduce on rows then columns:

    1 + 2 + 3 + 4
    +   +   +   +
    5 + 6 + 7 + 8
    
    = 
    6 + 8 + 10 + 12  = 36


Many operations can be expressed as reductions. These are **aggregate** operations.

`np.any` and `np.all` test if an array of Boolean values is all True or not all False \(i.e. if any element is True\). These are one kind of **aggregate function** -- a function that processes an array and returns a single value which "summarises" the array in some way.

* `np.any` is the reduction with logical OR
* `np.all` is the reduction with logical AND
* `np.min` is the reduction with min\(a,b\)
* `np.max` is the reduction with max\(a,b\)
* `np.sum` is the reduction with +
* `np.prod` is the reduction with *

Some functions are built on top of reductions:  
* `np.mean` is the sum divided by the number of elements reduced
* `np.std` computes the standard deviation using the mean, then some elementwise arithmetic

* By default, aggregate functions operate over the whole array, regardless of how many dimensions it has. 
* This means reducing over the last axis, then reducing over the second last axis, and so on, until a single scalar remains. 
* For example, `np.max(x)`, if `x` is a 2D array, will compute the reduction across columns and get the max for each row, then reduce over rows to get the max over the whole array.

We can specify the specific axes to reduce on using the `axes=` argument to any function that reduces.

### Accumulation

The sum of an array is a single scalar value. The **cumulative sum** or **running sum** of an array is an array of the same size, which stores the result of summing up every element until that point. 

This is almost the same as reduction, but we keep intermediate values during the computation, instead of collapsing to just the final result. The general process is called **accumulation** and it can be used with different operators.

For example, the accumulation of `[1,2,3,4]` with `+` is `[1, 1+2, 1+2+3, 1+2+3+4] = [1,3,6,10]`.

* `np.cumsum` is the accumulation of `+`
* `np.cumprod` is the accumulation of `*`
* `np.diff` is the accumulation of `-` \(but note that it has one less output than input\)

Accumulations operate on a single axis at a time, and you should specify this if you are using them on an array with more than one dimension \(otherwise you will get the accumulation of flattened array\). 


### Finding

There are functions which find **indices** that satisfy criteria. For example, the largest value along some axis.

* `np.argmax()` finds the index of the largest element
* `np.argmin()` finds the index of the smallest element
* `np.argsort()` finds the indices that would sort the array back into order
* `np.nonzero()` finds indices that are non-zero \(or True, for Boolean arrays\)

Finding indices is of great importance, because it allows us to cross\-reference across axes or arrays. For example, we can find the row where some value is maximised and then find the attribute which corresponds to it.

### Argsorting

**argsort** finds the indices that would put an array in order. It is an *extremely* useful operation.

## Week 2
### Algebra: a loss of structure

The **algebraic properties** of operators on real numbers \(associativity, distributivity, and commutativity\) are *not* preserved with the representation of numbers that we use for computations. The approximations used to store these numbers efficiently for computational purposes means that:

ab ≠ ba, a \+ b ≠ b \+ a, etc.
 
a\(b\+c\) ≠ ab \+ bc


### Number types

There are different representations for numbers that can be stored in arrays. Of these, **integers** and **floating point numbers** are of most relevance. Integers are familiar in their operation, but floats have some subtlety.

#### Integers

Integers represent whole numbers \(no fractional part\). They come in two varieties: signed and unsigned. In memory, these are \(normally!\) stored as binary 2's complement \(for signed\) or unsigned binary \(for unsigned\).

Most 64 bit systems support operations on at least the following integer types:

| name   | bytes | min                        | max                        |
|--------|-------|----------------------------|----------------------------|
| int8   | 1     | -128                       | 127                        |
| uint8  | 1     | 0                          | 255                        |
| int16  | 2     | -32,768                    | 32,767                     |
| uint16 | 2     | 0                          | 65,535                     |
| int32  | 4     | -2,147,483,648             | 2,147,483,647              |
| uint32 | 4     | 0                          | 4,294,967,295              |
| int64  | 8     | -9,223,372,036,854,775,808 | +9,223,372,036,854,775,807 |
| uint64 | 8     | 0                          | 18,446,744,073,709,551,615 |

An operation which exceeds the bounds of the type results in **overflow**. Overflows have behaviour that may be undefined; for example adding 8 to the int8 value 120 \(exceeds 127; result might be 127, or -128, or some other number\). In most systems you will ever see, the result will be to wrap around, computing the operation modulo the range of the integer type.

#### Floats

##### Floating point representation

Integers have \(very\) limited range, and don't represent fractional parts of numbers. Floating point is the most common representation for numbers that may be very large or small, and where fractional parts are required. Most modern computing hardware supports floating point numbers directly in hardware.

\(side note: floating point isn't **required** for fractional representation; *fixed point* notation can also be used, but is much less flexible in terms of available range. It is often faster on simple hardware like microcontrollers; however modern CPUs and GPUs have extremely fast floating point units\)

**All floating point numbers are is a compact way to represent numbers of very large range, by allowing a fractional number with a standardised range (*mantissa*, varies from 1.0 to just less than 2.0) with a scaling or stretching factor \(*exponent*, varies in steps of powers of 2\).**

*Floating point numbers can be thought of as numbers between 1.0 and 2.0, that can be shifted and stretched by doubling or halving repeatedly*

The advantage of this is that the for a relatively small number of digits, a very large range of numbers can be represented. The precision, however, is variable, with very precise representation of small numbers \(close to zero\) and coarser representation of numbers far from 0.

##### Sign, exponent, mantissa

A floating point number is represented by three parts, each of which is in practice an integer. Just like scientific notation, the number is separated into an exponent \(the magnitude of a number\) and a mantissa \(the fractional part of a number\).

##### Binary floating point

In binary floating point, calculations are done base 2, and every number is split into three parts. These parts are:

* the **sign**; a single bit indicating if a number is positive or negative
* the **exponent**; a signed integer indicating how much to "shift" the mantissa by
* the **mantissa**; an unsigned integer representing the fractional part of the number, following the 1.

A floating point number is equal to:

    sign * (1.[mantissa]) * (2^exponent)

##### The leading one    

Note that a leading 1 is inserted before the mantissa; this is because it is unnecessary to represent the first digit, as we know the mantissa represents a number between 1.0 \(inclusive\) and 2.0 \(exclusive\). Instead, the leading one is *implicitly* present in all computations.


The mantissa is always a positive number, stored as an integer such that it would be shifted until the first digit was just after the decimal point. So a mantissa which was stored as a 23 digit binary integer $00100111010001001000101_2$ would really represent the number $1.00100111010001001000101_2$
The exponent is stored as a positive integer, with an implied "offset" to allow it to represent negative numbers. 

For example, in `float32`, the format is:

    1     8      23
    sign  exp.   mantisssa
    
The exponents in `float32` are stored with an implied offset of -127 \(the "bias"\), so exponent=0 really means exponent=\-127.

So if we had a `float32` number

      1 10000011 00100111010001001000101
      
What do we know? 

1. The number is negative, because leading bit \(sign bit\) is 1.
1. The mantissa represents $1.00100111010001001000101_2 = 1.153389573097229_\{10\}$
1. The exponent represents $2^\{131-127\} = 2^4 = 16$ \($1000011_2=131_\{10\}$\), because of the implied offset.


##### IEEE 754

The dominant standard for floating point numbers is IEEE754, which specifies both a representation for floating point numbers and operations defined upon them, along with a set of conventions for "special" numbers.

The IEEE754 standard types are given below:

| Name       | Common name         | Base | Digits | Decimal digits | Exponent bits | Decimal E max | Exponent bias  | E min   | E max   | Notes     |
|------------|---------------------|------|--------|----------------|---------------|---------------|----------------|---------|---------|-----------|
| binary16   | Half precision      | 2    | 11     | 3.31           | 5             | 4.51          | 2^4−1 = 15      | −14     | +15     | not basic |
| **binary32**   | Single precision    | 2    | 24     | 7.22           | 8             | 38.23         | 2^7−1 = 127     | −126    | +127    |           |
| **binary64**   | Double precision    | 2    | 53     | 15.95          | 11            | 307.95        | 2^10−1 = 1023   | −1022   | +1023   |           |
| binary128  | Quadruple precision | 2    | 113    | 34.02          | 15            | 4931.77       | 2^14−1 = 16383  | −16382  | +16383  |           |
| binary256  | Octuple precision   | 2    | 237    | 71.34          | 19            | 78913.2       | 2^18−1 = 262143 | −262142 | +262143 | not basic |


#### Floats, doubles 

Almost all floating point computations are either done in **single precision** \(**float32**, sometimes just called "float"\) or **double precision** \(**float64**, sometimes just called "double"\). 

##### float32

**float32** is 32 bits, or 4 bytes per number; **float64** is 64 bits or 8 bytes per number.

GPUs typically are fastest \(by a long way\) using **float32**, but can do double precision **float64** computations at some significant cost. 

#### float64

**float64** is 64 bits, or 8 bytes per number. Most desktop CPUs \(e.g. x86\) have specialised **float64** hardware \(or for x86 slightly odd 80\-bit "long double" representations\).

#### Exotic floating point numbers

Some GPUs can do very fast **float16** operations, but this is an unusual format outside of some specialised machine learning applications, where precision isn't critical. \(there is even various kinds of **float8** used occasionally\).

**float128** and **float256** are very rare outside of astronomical simulations where tiny errors matter and scales are very large. For example, [JPL's ephemeris of the solar system](https://ssd.jpl.nasa.gov/?ephemerides) is computed using `float128`. Software support for `float128` or `float256` is relatively rare. NumPy does not support `float128` or `float256`, for example (it seems like it does, but it doesn't).

IEEE 754 also specifies **floating-point decimal** formats that are rarely used outside of specialised applications, like some calculators.

#### Binary representation of floats

We can take any float and look at its representation in memory, where it will be a fixed length sequence of bits \(e.g. float64 = 64 bits\). This can be split up into the sign, exponent and mantissa. 

#### Integers in floats

For **float64**, every integer from $-2^{53}$ to $2^{53}$ is precisely representable; integers outside of this range are not represented exactly \(this is because the mantissa is effectively 53 bits, including the implicit leading 1\).


#### Float exceptions

Float operations, unlike integers, can cause *exceptions* to happen during calculations. These exceptions occur at the *hardware* level, not in the operating system or language. The OS/language can configure how to respond to them \(for example, Unix systems send the signal SIGFPE to the process which can handle it how it wishes\).

There are five standard floating point exceptions that can occur.

* **Invalid Operation**  
Occurs when an operation without a defined real number result is attempted, like 0.0 / 0.0 or sqrt\(\-1.0\).

*  **Division by Zero**  
Occurs when dividing by zero.

* **Overflow**  
Occurs if the result of a computation exceeds the limits of the floating point number \(e.g. a `float64` operations results in a number \> 1e308\)

* **Underflow**  
Occurs if the result of a computation is smaller than the smallest representable number, and so is rounded off to zero.

* **Inexact**  
Occurs if a computation will produce an inexact result due to rounding.

----

Each exception can be **trapped** or **untrapped**. An untrapped exception will not halt execution, and will instead do some default operation \(e.g. untrapped divide by zero will output infinity instead of halting\). A trapped exception will cause the process to be signaled in to indicate that the operation is problematic, at which point it can either halt or take another action.

Usually, `invalid operation` is trapped, and `inexact` and `underflow` are not trapped. `overflow` and `division by zero` may or may not be trapped. NumPy traps all except `inexact`, but normally just prints a warning and continues; it can be configured to halt and raise an exception instead.


#### Special numbers: zero, inf and NaN
##### Zero: \+0.0 and \-0.0

IEEE 754 has both positive and negative zero representations. Positive zero has zero sign, exponent and mantissa. Negative zero has the sign bit set.

Positive and negative 0.0 compare equal, and work exactly the same in all operations, except for the sign bit propagating.

##### Infinity: \+∞ and \-∞

IEEE 754 floating point numbers **explicitly** encode infinities. They do this using a bit pattern of all ones for the exponent, and all zeros for the mantissa. The sign bit indicates whether the number is positive or negative.

NaN or **Not A Number** is a particularly important special "number". NaN is used to represent values that are invalid; for example, the result of 0.0 / 0.0. All of the following result in NaN:  
* `0 / 0`
* `inf / inf` \(either positive or negative inf\)
* `inf - inf` or `inf + -inf`
* `inf * 0` or  `0 * -inf`
* `sqrt(x)`, if x<0
* `log(x)`, if x<0
* Any other operation that would have performed any of these calculations internally

NaN has several properties:

* it **propagates**: any floating point operation involving NaN has the output NaN. \(almost: `1.0**nan==1.0`\).
* any comparison with NaN evaluates to false. NaN is not equal to anything, **including itself**; nor is it greater than or lesser than any other number. It is the only floating point number not equal to itself.
* NaN, however, is *not* equivalent to False in Python

It is both used as the *output of operations* \(to indicate where something has gone wrong\), and deliberately as a *placeholder in arrays* \(e.g. to signal missing data in a dataset\). 

NaN has all ones exponent, but non\-zero mantissa. Note that this means there is *not* a unique bit pattern for NaN. There are $2^{52}-1$ different NaNs in `float64` for example, all of which behave the same.


#### NaN as a result

It is very common experience to write some numerical code, and discover that the result is just NaN. This is because NaN propagates -- once it "infects" some numerical process, it will spread to all future calculations. This makes sense, since NaN indicates that no useful operation can be done. However, it can be a frustrating experience to debug NaN sources. 

The most common cause is **underflow** rounding a number to 0 or **overflow** rounding a number to \+/\-`inf`, which then gets used in one of the "blacklisted" operations.

#### NaN as a mask

Sometimes NaN is used to mask parts of arrays that have missing data. While there is specialised support for masked arrays in some languages/packages, NaNs are available everywhere and don't require any special storage or data structures.

For example, plotting data with NaN's in it results in gaps.


#### Roundoff and precision

Floating point operations can introduce **roundoff error**, because the operations involved have to quantize the results of computations. This can be subtle, because the precision of floating point numbers is variable according to their magnitude; unlike integers, where roundoff is always to the nearest whole number.

##### Repeated operations

This can be a real problem when doing repeated operations. Imagine adding dividends to a bank account every second \(real time banking!\)

#### Roundoff error

While this might seem an unlikely scenario, adding repeated tiny offsets to enormous values is exactly the kind of thing that happens in lots of simulations; for example, plotting the trajectory of a satellite with small forces from solar wind acting upon it.

The ordering of operations can be important, with the general rule being to avoid operations of numbers of wildly different magnitudes. 
**This means, in some extreme cases, that the distributive and associative rules don't apply to floating point numbers!**


#### Laws of floating-point disaster  
Some basic rules:  
1. `x+y` will have large error if x and y have different magnitudes \(magnitude error\)
1. `x-y` will have large error if `x~=y` \(cancellation error\)

#### Don't compare floats with ==

Because of the roundoff errors in floating point numbers, whenever there are comparisons to be made between floating point numbers, it is not appropriate to test for precise equality


### Array data structure

Specific implementations differ on exactly how they implement arrays. But a common feature is that the data in the arrays is tightly packed in memory, such that the memory used by an array is a small, constant overhead over the storage of the numbers that make up its elements. There is a short header which describes how the array is laid out in memory, followed by the numerical data.

Efficiency is achieved by packing in the numbers in one long, flat sequence, one after the other. Regardless of whether there is a 1D vector or a 5D tensor, the storage is just a sequence of numbers with a header at the start. 

We can see this flat sequence using `np.ravel()`, which "unravels" an array into the elements as a 1D vector; in reality it is just returning the array as it is stored in memory. \(caveat: this isn't quite the order in memory by default\).


#### Strides and shape

To implement multidimensional indexing, a key property of arrays,  the standard "trick" is to use **striding**,  with a set of memory offset constants \("strides"\) which specify how to index into the array, one per axis. This lets the system efficiently index into the array as if it were multidimensional, while keeping it as a single long sequence of numbers packed tightly into memory.

There is one **stride** per dimension, and each stride tells the system how many **bytes** forward to seek to find the next element *for each dimension* \(it's called a "stride" because it's how big of a step in memory to take to get to the next element\).
For a 1D array, there is one stride, which will be the length of the numeric data type \(e.g. 8 bytes for a standard float64\).

For a 2D array x, the first element might be 8 \(one float\), and the second might be `8*x.shape[0]`. In other words, to move to the next column, add 8; to move to the next row, add 8 times the number of columns. Strides are usually given in bytes, not elements, to speed up memory access computations.


### Dope fiends

This type of representation is correctly called a **dope vector**, where the **dope vector** refers to the striding information. It is held separately from the data itself; a header which specifies how to index.

The alternative is an **Illife vector**, which uses nested pointers to refer to multidimensional arrays. This is what happens, for example, if we make a list of lists in Python:

    [[1,2,3], [4,5,6], [7,8,8]]
    
The outer list refers to three inner lists.

**Illife vectors** can store ragged arrays trivially (no rectangularity requirement), but are much less efficient for large numerical operations that a form using **dope vectors**.

#### Java Illife vector

    int[][] a = new int[8][8];
    elt_3_4 = a[3][4]

#### Java dope vector

    int [] a = new a[64];
    row_offset = 8;
    elt_3_4 = a[row_offset*3 + 4];

#### A sketch of an array data structure

A typical structure might look like this. This is written as a C struct, but you can imagine equally well as a Java or Python class. This structure assumes the `dtype` is fixed to `float64` \(`double` in C\).

    // assume uint is a suitable unsigned integer type,
    // e.g. uint64_t 
    // assume double is IEEE754 float64
    
    struct NDArray        // assumes numbers are all doubles
    {
    
        uint n_dim;       // number of dimensions
        uint n_items;     // total number of elements
        uint item_size;   // size of one element  (in bytes)        
        uint *shape;      // shape, sequence of unsigned ints
        int *strides;     // striding, as a sequence of integers, 
                          // in *bytes* (may be negative!)
        
        double *data;     // pointer to the data array, 
                          // in this case float64 (double)
        
        uint flags;       // any special flags
    };


Most of these are self explanatory; e.g. for a 3x3 array of float64 holding all zeros we would have:

    n_dim = 2          // 2D, i.e. a matrix
    n_items = 9        // 3 * 3 = 9 elements
    item_size = 8      // 8 bytes per float64
    shape = {3,3}      // this is just the shape of the array
    strides = {8, 24}  // the offsets, in bytes, 
                       // to move to the next element in each dimension
    data = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} 
    // we'll ignore flags



#### Vectorised operations
Note that many operations can now be defined that just ignore the strides. For example, adding together two arrays:


      // some pseudo code
      if(same_shape(x,y))
      {
          z = zero_array_like(x);
          for(i=0;i<x->n_items;i++)          
                z->data[i] = x->data[i] + y->data[i];        
          return z;
      }


#### Strided arrays in practice

This is pretty much how NumPy arrays are implemented.

#### Transposing
This lets us do some clever things. For example, to transpose an array, all we need to do is to exchange the shape elements and the strides in the header. *The elements themselves are not touched, and the operation takes constant time regardless of array size*. Operations simply *read the memory in a different way* after the transpose.

Or we can reverse the array by setting the memory block pointer to point to the *end* of the array, and then setting the strides to be negative.

#### Rigid transformations

Rigid transformations of arrays like flipping and transposing do *not* change the data in the array. They only change the strides used to compute indexing operations. This means they do not depend on the size of the array, and are thus O\(1\) operations. \(technically they are O\(D\) where D is the number of dimensions of the array\)

### C and Fortran order

There is one thing you might have noticed: NumPy gave the strides of the original array as \[48,8\], *not* \[8,48\]. This is because the NumPy default \(and most common order generally\) is that the *last index changes first*.

This applies to any higher dimensional array as well: for example a \(512, 512, 3\) RGB color image in C order has the memory layout
    
    R -> G -> B -> R G B R G B -> going column-wise
    -> then row-wise


### Tensor operations

1D and 2D matrices are reasonably straightforward to work with. But many interesting data science problems like the EEG example involve arrays with higher rank; this is very common in deep learning, for example.

Being able to think about vectorised operations and then mold tensors to the right configurations to do what you want efficiently is a specialised and valuable skill.

#### Reshape

As well as "rigid" transformations like `np.fliplr()` and `np.transpose()`, we can also reshape arrays to completely different shapes. The requirement is that the *elements* don't change; **thus the total number of elements cannot change during a reshaping operation.**

#### Reshaping rules  
* The total number of elements will remain unchanged.
* The order of elements will remain unchanged; only the positions at which the array "wraps" into the next dimension.
* The last dimension changes fastest, the second last second fastest, etc.

##### The pouring rule  
You can imagine a reshaping operation "pouring" the elements into a new mold. The ordering of the elements is retained, but will fill up the new shape. Pouring fills up the last dimension first.


#### Squeezing

Singleton dimensions can get in the way of doing computations easily; often the result of a complex calculation is a multidimensional array with a bunch of singleton dimensions.

**squeezing** just removes *all* singleton dimensions in one go and is performed by `np.squeeze()`

#### Swapping and rearranging axes

We can rearrange the axes of our arrays as we wish, using `np.swapaxes(a, axis1, axis2)` to swap any pair of axes. For example, if we have a colour video, which is of shape `(frames, width,  height,  3)` and we want to apply an operation on each *column*, we can temporarily swap it to the end, broadcast, then swap it back. 


#### The swap, reshape, swap dance

Reshape always follows the pouring rule \(last dimension pours first\). Sometimes that isn't what we want to do. The solution is to:
* rearrange the axes
* reshape the array
* \(optionally\) rearrange the axes again

Imagine we want to get the cat gif as a film strip, splitting the colour channels into three rows, one for red, green and blue, as if we had three strips of celluloid film. This will be a reshape to size `(H*3, W*Frames)`.

### Einstein summation notation

A very powerful generalisation of these operations is **Einstein summation notation**. This, in its simplest form, is a very easy way to reorder higher-rank tensors. As you can see from the example above, swapping axes gets confusing very fast. Einstein summation notation allows a specification of one letter names for dimensions \(usually from `ijklmn...`\), and then to write the dimension rearrangement *as a string*.

The notation writes the original order, followed by an `->` arrow, then a new order.

    ijk -> jik

#### Extreme einsumming
`einsum` can also compute summations, products and diagonalisations in one single command

The power of `einsum` leverages the flexibility of the `ndarray` strided array structure. It makes a huge range of operations possible in a very compact form.

## Week 3


### What is visualisation?

Visualisation is the representation of data for human visual perception. Visualisation encodes abstract mathematical objects, like vectors of numbers, into a form that *humans* can draw meaning from. It encompasses everything from the most elementary bar chart to the most sophisticated volume renderings.

Visualisation serves several purposes:
* to build intuition about *relationships and structure* in data;
* to *summarise* large quantities of data;
* to help decision-makers make quick judgments on specific questions.


### Grammar of Graphics

The creation of scientific visualisations can be precisely specified in terms of a *graphical language*. This is a language which specifies how to turn data into images \(construction\) and how readers should interpret those images \(intepretation\). For the moment assume a **Dataset** is a table of numbers \(e.g. a 2D NumPy array\). Each column represents a different attribute; each row a collection of related attributes:

        year    price    margin
        -------------------------
        1990    0.01      0.005
        1991    0.018    -0.001
        1992    0.002    -0.1
        1993    0.02      0.005

There are various different ways of describing the graphical language used; the one described here follows the **Layered Grammar of Graphics**.


* **Stat**  
A **stat** is a statistic computed from data which is then mapped onto visual features, with the intent of summarising the data compactly. For example, the mean and the standard deviation of a series of measurements would be **stats**. The binning of values in a histogram is another example of a **stat**.

* **Mapping**  
A **mapping** represents a transformation of data attributes to visual values. It maps *selected attributes* (which includes **stats** or raw dataset values) to visual values using a **scale** to give units to the attribute.
    * **Scale** a scale specifies the transformation of the units in the dataset/stat to visual units. This might be from the Richter scale to x position, or from altitude to colour, or from condition number to point size. A scale typically specifies the **range** of values to be mapped.
    
    * **Guide**  
    A **guide** is a visual reference which indicates the meaning of a mapping, including the scale and the attribute being mapped. This includes axis tick marks and labels, colour scales, and legends.
    
* **Geom**  
A **geom** is the geometric representation of data after it has been mapped. Geoms include points (which may have size, colour, shape), lines (which may have colour, dash styles, thickness), and patches/polygons (which can have many attributes).

* **Coord**  
A **coord** is a coordinate system, which connects mapped data onto points on the plane (or, in general, to higher-dimensional coordinates, like 3D positions). The spatial configuration of **geoms** and **guides** depends on the coordinate system. 

* **Layer**   
A **layer** of a plot is one set of geoms, with one mapping on one coordinate system. Multiple layers may be overlaid on a single coordinate system. For example, two different stats of the same dataset might be plotted on separate layers on the same coordinate system.

* **Facet**  
A **facet** is a different view on the same dataset, on a separate coordinate system. For example, two conditions of an experiment might be plotted on two different facets. One facet might have several layers.

* **Figure**  
A **figure** comprises a set of one or more **facets**
* **Caption** Every figure has a caption, which explains the visualisation to the reader.


### Anatomy of a figure

#### Figure and caption
A **figure** is one or more facets that form a coherent visualisation of data. Many figures are single graphs \(single facets\), but some may include multiple facets. Every figure needs to have a clear **caption** which explains to the reader what they should see from the graph.

#### The visual representation of plots: guides, geoms and coords

Good scientific plots have a well defined structure. Every plot should have proper use of **guides** to explain the **mapping** and **coordinate system**:

#### Guides

* **Axes** which are labeled, with any applicable **units** shown. These are **guides** for the **coordinate system**.
    * **Ticks** indicating subdivisions of axes in labeled units for that axis.
* A **legend** explaining what markers and lines mean \(if more than one present\). These are **guides** which identifies different **layers** in the plot.
* A **title** explaining what the plot is

A plot might have:

* A **grid** to help the reader line up data with the axes \(again, a **guide** for the **coordinate system**\)
* A **annotations** to point out relevant features

#### Geoms

To display data, plots have **geoms**, which are geometrical objects representing some element of the data to be plotted. These include:

* **Lines/curves** representing continuous functions, which have colours, thicknesses, and styles 
* **Markers** representing disconnected points, which have sizes, colours and styles
* **Patches** representing shapes with area \(like bars in a bar chart\), which can come in many forms


### Basic  2D plots

#### Dependent and independent variables

Most *useful* plots only involve a small number of relations between variables. Often there are just two variables with a relation; one **independent** variable and one **dependent** variable, which depends on  the independent variable.

$$y = f(x),$$

where $x$ and $y$ are scalar variables. The purpose of the plot is to visually describe the function $f$.  The input to these plots are a *pair* of 1D vectors $\vec{x}, \vec{y}$. 


These are plotted with the **independent variable** on the x\-axis \(the variable that "causes" the relationship\) and the **dependent value** on the y\-axis \(the "effect" of the relationship\).  For example, if you are doing an experiment and fixing some value to various test values \(e.g. the temperature of a solution\) and measuring another \(e.g. the pH of the solution\), the variable being fixed is the independent variable and goes on the $x$ axis, and the variable being measured is the dependent variable and goes on the $y$ axis.

In some cases, there is no clear division between an the independent and dependent variable, and the order doesn't matter. This is unusual, however, and you should be careful to get the ordering right.

#### Common 2D plot types

There are a few common 2D plot types. As always, we are working with arrays of data. Our mapping takes *two columns* of a dataset \(or a stat of a dataset\) and maps one to $x$ and one to $y$.

#### Scatterplot

A scatter plot marks $(x,y)$ locations of measurements with *markers*. These are point **geoms**.

#### Bar chart

A bar chart draws bars with height proportional to $y$ at position given by $x$.  These are patch **geoms**.

#### Line plot

A line plot draws connected line segments between \(x,y\) positions, in the order that they are provided.  These are line **geoms**.

#### Marking measurements

It is common to plot the explicit points at which measurements are available. This is a plot with two **layers**, which therefore share the same **coords**. 
One layer represents the data with **line geoms**, the second represents the same data with **point geoms** \(markers\).

#### Ribbon plots

If we have a triplet of vector $\vec{x},\vec{y_2},\vec{y_1}$, where both $y$s match the $x$ then the area between the two lines can be drawn using polygon **geoms**. This results in a **ribbon plot**, where the ribbon can have variable thickness \(as the difference $y_1-y_2$ varies\).

This can be also be thought of a line geom with variable width along the $x$ axis.

#### Layering geoms

This often combined in a **layered** plot, showing the data with:  
* line geoms for the trend of the data
* point geoms for notating the actual measurements
* area geoms to represent measurement uncertainty 


#### Units

If an axes represents real world units \(**dimensional quantities**\), like  *millimeters, Joules, seconds, kg/month, hectares/gallon, rods/hogshead, nanometers/Coulomb/parsec* the units should *absolutely always* be specified in the axis labels. 

Use units that are of the appropriate scale. Don't use microseconds if all data points are in the month range. Don't use kilowatts if the data points are in nanowatts. There are exceptions to this if there is some standard unit that not using would be confusing. 



Only if the quantities are truly **dimensionless** \(**bare numbers**\), like in the graph of the pulses at the start of the lecture, should there be axes without visible units. In these cases the axis should be clearly labeled \(e.g. "Relative growth", "Mach number", etc.\).

Never use the index of an array as the $x$ axis unless it is the only reasonable choice. For example, if you have data measured in regular samples over time, give the units in real time elapsed \(e.g. microseconds\), not the sample number!

#### Avoid rescaled units

Although `matplotlib` and other libraries will autoscale the numbers on the axis to sensible values and then insert a `1eN` label at the top, you should avoid this. This can lead to serious confusion and is hard to read.

#### Axes and coordinate transform

An **axis** is used to refer to the visual representation of a dimension on a graph \(e.g. "the x axis"\), and the object which *transforms data from measurement units to visual units*. In other words, the axis specifies the scaling and offset of data: a coordinate system or **coord**. 

The mapping of data is determined by **axis limits** which specify the minimum and maximum measurement values to be displayed.


### Stats

A **stat** is a statistic of a **Dataset** \(i.e. a function of the data\). Statistics *summarise* data in some way. Common examples of stats are:

* **aggregate summary statistics**, like measures of central tendency \(mean, median\) and deviation \(standard deviation, max/min, interquartile range\)
* **binning operations**, which categorise data into a number of discrete **bins** and count the data points falling into those bins
* **smoothing and regression**, which find approximating functions to datasets, like linear regression which fits a line through data

Plots of a single 1D array of numbers $[x_1, x_2, \dots, x_n]$ usually involve displaying *statistics* \(**stats**\)  to transform the dataset into forms that 2D plotting forms can be applied to. Very often these plot types are used with multiple arrays \(data that has been grouped in some way\) to show differences between the groups.

#### Binning operations

A **binning** operation divides data into a number of discrete groups, or **bins**. This can help summarise data, particularly for continuous data where every observation will be different if measured precisely enough \(for example, a very precise thermometer will *always* show a different temperature to the same time yesterday\).

#### Histograms: showing distributions

A **histogram** is the combination of a **binning operation** \(a kind of **stat**\) with a standard 2D bar chart \(the bars being the **geoms**\). It  shows the count of values which fall into a certain range. It is useful when we want to visualise the *distribution* of values.

A histogram does not have spaces between bars, as the bin regions represent contiguous divisions of the input space.


### Ranking operations
#### Sorted bar chart

An alternative view of a 1D vector is a sorted chart or **rank plot**, which is a plot of a value against its rank within the array. The value-to-rank operation is the **stat** which is applied. 

### Aggregate summaries

Any standard summary statistic can be used in a plotting operation. A particularly common statistical transform is to represent ranges of data using either the minimum/maximum, the standard deviation, or the interquartile range. These are often ranges of **groupings** of data \(e.g. conditions of an experiment\).

#### Box plot

A **Box plot** \(this is actually named after George Box of Box-Cox fame, not because it looks like a box!\) is a visual summary of an array of values. It computes multiple **stats** of a dataset, and shows them in a single **geom** which has multiple components.
It is an extremely useful plot for comparing the *distribution* of values.

The value shown are usually:  
* the **interquartile range** \(the range between the 75% and 25% percentiles of the dataset\), shown as a box
* the **median** value, shown as a horizontal line, sometimes with a "notch" in the box at this point.
* the **extrema**, which can vary between plots, but often represent the 2.5% and 98.5% percentiles \(i.e. span 95% of the range\). They are drawn as *whiskers*, lines with a short crossbar at the end.  
* **outliers**, which are all dataset points outside of the extrema, usually marked with crosses or circles.

#### Violin plot

The **violin plot** extends the Box plot to show the distribution of data more precisely. Rather than just a simple box, the full distribution of data is plotted, using a smoothing technique called a **kernel density estimate**.


### Regression and smoothing

Regression involves finding an **approximating** function to some data; usually one which is *simpler* in some regard. The most 
familiar is **linear regression** -- fitting a line to data. 

$$ f(x) = mx + c,\   f(x) \approx y $$

We will not go into how such fits are computed here, but they are an important class of **stats** for proposing hypotheses which might *explain* data.

#### Smoothing

Likewise, the moving average smoother we saw in the earlier gas example finds a "simplified" version of the data by removing fast changes in the values. There are many ways of smoothing data, all of which imply some assumption about how the data should behave. A good smoothing of data reveals the attributes of interest to the reader and obscures irrelevant details.

### Geoms

#### Markers

Markers are *geoms* that represent bare points in a coordinate system. These typically as a visual record of an discrete **observation**. In the simplest case, they are literally just points marked on a graph; however, they typically convey more information.
There are several important uses of markers:

#### Layer identification

The geom used for marking points can be used to differentiate different layers in a plot. Both the shape and the colouring can be modified to distinguish layers. Excessive numbers of markers quickly become hard to read!

#### Colour choices

Choosing good colour for plots is essential in comprehension. Care should be taken if plots may be viewed in black and white printouts, where colour differences will become differences in shade alone.  Also, a significant portion of the population suffers from some form of colour blindness, and pure colour distinctions will not be visible to these readers.

#### Markers: Scalar attributes

Markers can also be used to display a *third* or event *fourth* scalar attribute, instead of identifying layers in a plot. In this case we, are not visualising just a pair of vectors $\vec{x}, \vec{y}$, but $\vec{x}, \vec{y}, \vec{z}$ or possibly $\vec{x}, \vec{y}, \vec{z}, \vec{w}$. This is done by modulating the *scaling* or *colouring* of each marker. 

#### Colour maps

Colouring of markers is done via a **colour map**, which maps scalar values to colours. 

Colour maps should always be presented with a **colour bar** which shows the mapping of values to colours. This is an example of a **guide** used for an aesthetic **mapping** beyond the 2D coordinate system.

#### Unsigned scalar

* If the data to be represented is a positive scalar \(e.g. heights of people\), use a colour map with **monotonically varying brightness**. This means that as the attribute increases, the colour map should get consistently lighter or darker.  `viridis` is good, as is `magma`. These are **perceptually uniform**, such that a change in value corresponds to a perceptually uniform change in intensity across the whole scale \(the human visual system is very nonlinear, and these scales compensate for this nonlinearity\). Grayscale or monochrome maps can be used, but colours with brightness+hue are often easier to interpret.

* **monotonic brightness** increasing data value always leads to an increase in visual brightness
* **perceptually uniform** a constant interval increase in data value leads to a perceptually constant increase in value.

#### Perceptual linearity

**Colormaps shown in grayscale. Good scales should be monotonic \(i.e. always increase/decrease in brightness\)**

#### Scaling colourmaps

Scale data to colorscales appropriately, and always provide a colour bar for reference. It **must** be possible to invert the visual units to recover the data units, and the colour bar is the essential **guide** for that purpose.

### Lines

Lines are *geoms* that connect points together. A line should be used if it makes sense to ask what is *between* two data points; i.e. if there is a continuum of values. This makes sense if there are two *ordered* arrays which represent samples from a *continuous* function 

$$y=f(x).$$

#### Linestyles

Line geoms can have
**variable thickness and variable colour**

They may also have different **dash patterns** which make it easy\(ish\) to distinguish different lines  without relying on colour. Colour may not be available in printed form, and excludes readers with colour blindness.

More than four dash patterns on one plot is bad idea, both from an aesthetic and a communication stand point.

#### The staircase and the bar

In some cases, it makes sense to join points with lines \(because they form a continuous series\), but we know that the value cannot have changed between measurements. For example, imagine a simulation of a coin toss experiment, plotting the cumulative sum of heads seen. 

### Alpha and transparency

Geoms can be rendered with different **transparency**, so that geoms layered behind show through. This is referred to as **opacity** \(the inverse of transparency\) or the **alpha** \(equivalent to opacity\). This can be useful when plotting large numbers of geoms which overlap \(e.g. on a dense scatterplot\), or to emphasise/deemphasise certain geoms, as with line thickness.

Transparency should be used judiciously; a plot with many transparent layers may be hard to interpret, but it can be a very effective way of providing visual emphasis.


### Coords

So far, we have assumed a simple model for coordinate systems \(**coords**\). We have just mapped two dimensions onto a two dimensional image with some scaling. More precisely, we have assumed that the mapping from data to visual units is linear mapping of each independent data dimension $x$ and $y$ to a Cartesian coordinate frame defined by a set of **axis limits**.

An axis limit specifies a range in **data units** \(e.g. 0 to 500 megatherms\) which are then mapped onto the available space in the figure in **visual units** \(e.g. 8cm or 800px\).

In `matplotlib` for example, we can control the visual units of a figure using `figsize` when creating a figure \(which, by default, are in inches\).

The axes will span some portion of that figure, and these define the **coord** for the visualisation. The axis limit `ax.set_xlim()` and `ax.set_ylim()` commands specify the data unit range which is mapped on. 

### Aspect ratio

In some cases, the **aspect ratio** of a visualisation is important. For example, images should never be stretched or squashed when displayed. 

### Coords in general

A coordinate system encompasses a **projection** onto the 2D plane, and might include transformations like polar coordinates, logarithmic coordinates or 3D perspective projection. We have so far seen only linear Cartesian coordinates, which are very simple projections. But there are other projections which can be useful to reveal structure in certain kinds of datasets.

### Log scales

Sometimes a linear scale is not an effective way to display data. This is particularly true when datasets have very large spans of magnitude. In this case, a **logarithmic** coordinate system can be used.

Log scales can be used either on the $x$-axis, $y$-axis or both, depending on which variable\(s\) have the large magnitude span.
If the plot is log in $y$ only, the plot is called "semilog $y$"; if $x$ only, "semilog $x$" and if logarithmic on both axes "log-log".

In `matplotlib`, `set_xscale`/`set_yscale` can be used to change between linear and log scales on each axis. \(side note: there are also commands `semilogx`, `semilogy` and `loglog` which are just aliases for `plot` with the appropriate scale change\).

#### Polynomial or power-law relationships

Log\-log scales \(log on both $x$ and $y$ axes\) are useful when there is a \(suspected\) polynomial relationship between variables \(e.g. $y=x^2$ or $y=x^\frac{1}{3}$\). The relationship will appear as a straight line on a log-log plot. 

$$ f(x) = x^k $$

looks linear if plot on `loglog`. The gradient of the line tells you the value of `k`.

#### Negative numbers

Note that log scales have one downside: the log of a negative number is undefined, and in fact the logarithm diverges to \-infinity at 0.

### Polar

Cartesian plots are the most familiar coords. But some data is naturally represented in different mappings. The most notable of these is **polar coordinates**, in which two values are mapped onto an *angle* $\theta$ and a *radius* $r$. 

This most widely used for data that originated from an angular measurement, but it can be used any time it makes sense for one of the axes to "wrap around" smoothly. The classic example is *radar data* obtained from a spinning dish, giving the reflection distance at each angle of the dish rotation. Similarly, wind data from weather stations typically records the direction and speed of the strongest gusts.

#### Negative numbers \(again\)

The sign issue raises it head again with polar plots -- there isn't a natural way to represent radii below zero on a polar plot. In general, polar plots should be reserved for mappings where the data mapping onto $r$ is positive.

### Facets and layers

We have seen several examples where multiple **geoms** have been used in a single visualisation. 

There are two ways in which this can be rendered:
* as distinct **layers** superimposed on the same set of **coords**
* as distinct **facets** on separate sets of **coords** \(with separate **scales** and **guides**\)

#### Layers

Layering is appropriate when two or more views on a dataset are closely related, and the data mapping are in the same units. For example, the historical wheat price data can be shown usefully as a layered plot. A **legend** is an essential **guide** to distinguish the different layers that are present. If multiple layers are used, a legend should \(almost\) always be present to indicate which **geom** relates to which dataset attribute.

### Facets

A much better approach is to use **facets**; separate **coords** to show separate aspects of the dataset. Facets have no need to show the same scaling or range of data. However, if two facets show the same attribute \(e.g. two facets showing shillings\) they should if possible have the same scaling applied to make comparisons easy.

#### Facet layout

There are many ways to lay out **facets** in a **figure**, but by far the most common is to use a regular grid. This is the notation that `matplotlib` uses in `fig.add_subplot(rows, columns, index)`. Calling this function creates a notional grid of `rows` x `columns` and then returns the axis object \(representing a **coord**\) to draw in the grid element indexed by `index`. Indices are counted starting from 1, running left to right then bottom to top. 

### Communicating Uncertainty

It is critical that scientific visualisations be **honest**; and that means representing **uncertainty** appropriately. Data are often collected from measurements with **observation error**, so the values obtained are corrupted versions of the true values. For example, the reading on a thermometer is not the true temperature of the air.

Other situations may introduce other sources of error; for example, roundoff error in numerical simulations; or uncertainty over the right choice of a mathematical model or its parameters used to model data.

These must be represented so that readers can understand the uncertainty involved and not make judgements unsupported by evidence.

We have already seen the basic tools to represent uncertainty in plots: **stats** like the standard deviation or interquartile range can be used to show summaries of a collection of data.

#### Error bar choice

There are several choices for the error bars:  
* the **standard deviation**
* the **standard error**
* **confidence intervals** \(e.g. 95%\)
* **nonparametric intervals** such as interquartile ranges.

#### Box plots

A very good choice for this type of plot is to use a **Box plot**, as we saw earlier. This makes clear the spread of values that were encountered in a very compact form.

#### Dot plot

An alternative is simply to jitter the points on the x\-axis slightly.

## Week 4

### Vector spaces
In this course, we will consider vectors to be ordered tuples of real numbers $[x_1, x_2, \dots x_n], x_i \in \mathbb{R}$ (the concept generalises to complex numbers, finite fields, etc. but we'll ignore that). A vector has a fixed dimension $n$, which is the length of the tuple. We can imagine each element of the vector as representing a distance in an **direction orthogonal** to all the other elements.

For example, a length-3 vector might be used to represent a spatial position in Cartesian coordinates, with three orthogonal measurements for each vector. Orthogonal just means "independent", or, geometrically speaking "at 90 degrees".


* Consider the 3D vector [5, 7, 3]. This is a point in $\real^3$, which is formed of:

            5 * [1,0,0] +
            7 * [0,1,0] +
            3 * [0,0,1]
            
Each of these vectors [1,0,0], [0,1,0], [0,0,1] is pointing in a independent direction (orthogonal direction) and has length one. The vector [5,7,3]
can be thought of a weighted sum of these orthogonal unit vectors (called **"basis vectors"**). The vector space has three independent bases, and so is three dimensional.

We write vectors with a bold lower case letter:
$$\vec{x} = [x_1, x_2, \dots, x_d],\\
\vec{y} = [y_1, y_2, \dots, y_d],$$ and so on.

#### Points in space

##### Notation: $\real^n$

* $\real$ means the set of real numbers.  
* $\real_{\geq 0}$ means the set of non-negative reals.  
* $\real^n$ means the set of tuples of exactly $n$ real numbers. 
* $\real^{n\times m}$ means the set of 2D arrays (matrix) of real numbers with exactly $n$ rows of $m$ elements.

* The notation $(\real^n, \real^n) \rightarrow \real$ says that than operation defines a map from a pair of $n$ dimensional vectors to a real number.

##### Vector spaces

Any vector of given dimension $n$ lies in a **vector space**, called $\real^n$ (we will only deal with finite-dimensional real vector spaces with standard bases), which is the set of possible vectors of length $n$ having real elements, along with the operations of:   
*  **scalar multiplication** so that $a{\bf x}$  is defined for any scalar $a$. For real vectors, $a{\bf x} = [a x_1, a x_2, \dots a x_n]$, elementwise scaling.
    * $(\real, \real^n) \rightarrow \real^n$
* **vector addition** so that ${\bf x} + {\bf y}$ vectors ${\bf x, y}$ of equal dimension. For real vectors, ${\bf x} + {\bf y} = [x_1 + y_1, x_2 + y_2, \dots x_d + y_d]$ the elementwise sum
    * $(\real^n, \real^n) \rightarrow \real^n$


We will consider vector spaces which are equipped with two additional operations:  
* a **norm** $||{\bf x}||$ which allows the length of vectors to be measured.
    * $\real_n \rightarrow \real_{\geq 0}$
* an **inner product** $\langle {\bf x} | {\bf y} \rangle$ or ${\bf x \bullet y}$  which allows the angles of two vectors to be compared. The inner product of two orthogonal vectors is 0. For real vectors ${\bf x}\bullet{\bf y} = x_1 y_1 + x_2 y_2 + x_3 y_3 \dots x_d y_d$
    * $(\real^n, \real^n) \rightarrow \real$

All operations between vectors are defined within a vector space. We cannot, for example, add two vectors of different dimension, as they are elements of different spaces.

##### Topological and inner product spaces

With a norm a vector space is a **topological vector space**. This means that the space is continuous, and it makes sense to talk about vectors being "close together" or a vector having a neighbourhood around it. With an inner product, a vector space is an **inner product space**, and we can talk about the angle between two vectors.

##### Are vectors points in space, arrows pointing from the origin, or tuples of numbers?

These are all valid ways of thinking about vectors. Most high school mathematics uses the "arrows" view of vectors. Computationally, the tuple of numbers is the *representation* we use. The "points in space" mental model is probably the most useful, but some operations are easier to understand from the alternative perspectives. 

The points mental model is the most useful *because* we tend to view:  
* vectors to represent *data*; data lies in space
* matrices to represent *operations* on data; matrices warp space.

### Relation to arrays

These vectors of real numbers can be represented by the 1D floating point arrays we called "vectors" in the first lectures of this series. But be careful; the representation and the mathematical element are different things, just as floating point numbers are not real numbers.

## Uses of vectors

Vectors, despite their apparently simple nature, are enormously important throughout data science. They are a *lingua franca* for data. Because vectors can be  
* **composed** \(via addition\), 
* **compared** \(via norms/inner products\) 
* and **weighted** \(by scaling\), 

they can represent many of the kinds of transformations we want to be able to do to data.

On top of this, they map onto the efficient **ndarray** data structure, so we can operate on them efficiently and concisely.

### Vector data

Datasets are commonly stored as 2D **tables**. These can be seen as lists of vectors. Each **row** is a vector representing an "observation" \(e.g. the fluid flow reading in 10 pipes might become a 10 element vector\). Each observation is then stacked up in a 2D matrix. Each **column** represents one element of the vector across many observations.

We have seen many datasets like this \(synthetic\) physiological dataset:

    heart_rate systolic diastolic vo2 
     67           110    72       98
     65           111    70       98
     64           110    69       97
     ..
     
Each **row** can be seen as a vector in $\real^n$ \(in $\real^4$ for this set of physiological measurements\). The whole matrix is a sequence of vectors in the same vector space. This means we can make **geometric** statements about tabular data.

### Geometric operations

Standard transformations in 3D space include:

* scaling
* rotation
* flipping \(mirroring\)
* translation \(shifting\)

as well as more specialised operations like color space transforms or estimating the surface normals of a triangle mesh \(which way the triangles are pointing\). 

GPUs evolved from devices designed to do these geometric transformations extremely quickly. A vector space formulation lets all geometry have a common representation, and *matrices* \(which we will see later\) allow for efficient definition of operations on portions of that geometry.

Graphical pipelines process everything \(spatial position, surface normal direction, texture coordinates, colours, and so on\) as large arrays of vectors. Programming for graphics on GPUs largely involves packing data into a low-dimensional vector arrays \(on the CPU\) then processing them quickly on the GPU using a **shader language**.

Shader languages like HLSL and GLSL have special data types and operators for working with low dimensional vectors:

    # some GLSL
    vec3 pos = vec3(1.0,2.0,0.0);
    vec3 vel = vec3(0.1,0.0,0.0);
    
    pos = pos + vel;

### Machine learning applications

Machine learning relies heavily on vector representation. A typical machine learning process involves:

* transforming some data onto **feature vectors** 
* creating a function that transforms **feature vectors** to a prediction \(e.g. a class label\)

The **feature vectors** are simply an encoding of the data in vector space, which could be as simple as the tabular data example above, and feature transforms \(the operations that take data in its "raw" form and output feature vectors\) range from the very simple to enormously sophisticated. 

*Most machine learning algorithms can be seen as doing geometric operations: comparing distances, warping space, computing angles, and so on.*

One of the simplest effective machine learning algorithms is **k nearest neighbours**. This involves some *training set* of data, which consists of pairs $\vec{x_i}, y_i$: a feature vector $\vec{x_i}$ and a label $y_i$. 

When a new feature needs classified to make a prediction, the $k$ *nearest* vectors in this training set are computed, using a **norm** to compute distances. The output prediction is the class label that occurs most times among these $k$ neighbours \($k$ is preset in some way; for many problems it might be around 3\-12\).

The idea is simple; nearby vectors ought to share common properties. So to find a property we don't know for a vector we do know, look at the properties that nearby vectors have.

### Image compression

Images have a straightforward representation as 2D arrays of brightness, as we have seen already. But, just like text, this representation is rather empty in terms of the operations that can be done to it. A single pixel, on its own, has as little meaning as a single letter.

Groups of pixels -- for example, rectangular patches -- can be unraveled into a vector. An 8x8 image patch would be unraveled to a 64\-dimensional vector. These vectors can be treated as elements of a vector space.

Many image compression algorithms take advantage of this view. One common approach involves splitting images into patches, and treating each patch as a vector $\vec{x_1}, \dots, \vec{x_n}$. The vectors are **clustered** to find a small number of vectors $\vec{y_1}, \dots, \vec{y_m},\ m << n$ that are a reasonable approximation of nearby vectors. Instead of storing the whole image, the vectors for the small number of representative vectors $\vec{y_i}$ are stored \(the **codebook**\), and the rest of the image is represented as the *indices* of the "closest" matching vector in the codebook i.e. the vector $\vec{y_j}$ that minimises $||x_i - y_j||$. 

This is **vector quantisation**, so called because it quantises the vector space into a small number of discrete regions. This process maps **visual similarity onto spatial relationships.**

### Basic vector operations

There are several standard operations defined for vectors, including getting the length of vectors,  and computing dot (inner), outer and cross products.

#### Addition and multiplication

Elementwise addition and scalar multiplication on arrays already implement the mathematical vector operations. Note that these ideas let us form **weighted sums** of vectors:  
$$\lambda_1 \vec{x_1} + \lambda_2 \vec{x_2} + \dots + \lambda_n \vec {x_n}$$

This applies **only** to vectors of the same dimension.

### How big is that vector?

Vector spaces do not necessarily have a concept of distance, but the spaces we will consider can have a distance *defined*. It is not an inherent property of the space, but something that we define such that it gives us useful measures.

The Euclidean length of a vector $\bf x$ (written as $||{\bf x}||$) can be computed directly using `np.linalg.norm()`. This is equal to:

$$ \|{\bf x}\|\_2 = \sqrt{x_0^2 + x_1^2 + x_2^2 + \dots + x_n^2  } $$

and corresponds to the radius of a \(hyper\)sphere that would just touch the position specified by the vector.

### Different norms

The default `norm` is the **Euclidean norm** or **Euclidean distance measure**; this corresponds to the everyday meaning of the word "length". A vector space of real vectors with the Euclidean norm is called a **Euclidean space**. The distance between two vectors is just the norm of the difference of two vectors: $$||{\bf x-y}||\_2$$ is the distance from $\bf x$ to $\bf y$

But there are multiple ways of measuring the length of a vector, some of which are more appropriate in certain contexts. These include the $L_p$-norms or Minkowski norms, which generalise Euclidean distances, written $$
\|{\bf x}\|_p$$.

| p         | Notation              | Common name                    | Effect                 | Uses                                          | Geometric view                     |
|-----------|-----------------------|--------------------------------|------------------------|-----------------------------------------------|------------------------------------|
| 2         | $\|x\|$ or $\|x\|_2$  | Euclidean norm                 | Ordinary distance      | Spatial distance measurement                  | Sphere just touching point         |
| 1         | $\|x\|_1$             | Taxicab norm; Manhattan norm   | Sum of absolute values | Distances in high dimensions, or on grids     | Axis-aligned steps to get to point |
| 0         | $\|x\|_0$             | Zero pseudo-norm; non-zero sum | Count of non-zero values | Counting the number of "active elements"    | Numbers of dimensions not touching axes                                |
| $\infty$  | $\|x\|_\inf$          | Infinity norm; max norm        | Maximum element        | Capturing maximum "activation" or "excursion" | Smallest cube enclosing point      |
| $-\infty$ | $\|x\|_{-\inf}$       | Min norm                       | Minimum element        |    Capturing minimum excursion                                           |                         Distance of point to closest axis           |  

### Unit vectors and normalisation

A unit vector has norm 1 (the definition of a unit vector depends on the norm used). Normalising for the Euclidean norm can by done by scaling the vector ${\bf x}$ by $\frac{1}{||{\bf x}||\_2}$. A unit vector nearly always refers to a vector with Euclidean norm 1.

If we think of vectors in the physics sense of having a **direction** and **length**, a unit vector is "pure direction". If normalised using the $L_2$ norm, for example, a unit vector always lies on the surface of the unit sphere.

### Inner products of vectors

An inner product $(\real^N \times \real^N) \rightarrow \real$ measures the *angle* between two real vectors. It is related to the **cosine distance**

The inner product is only defined between vectors of the same dimension, and only in inner product spaces. 

The inner product is a useful operator for comparing vectors that might be of very different magnitudes, since it does not depend on the magnitude of the vectors, just their directions. For example, it is widely used in information retrieval to compare **document vectors** which represent terms present in a document as large, sparse vectors which might have wildly different magnitudes for documents of different lengths.

### Basic vector statistics

Given our straightforward definition of vectors, we can define some  **statistics** that generalise the statistics of ordinary real numbers. These just use the definition of vector addition and scalar multiplication, along with the outer product.

The **mean vector** of a collection of $N$ vectors is the sum of the vectors multiplied by $\frac{1}{N}$:

$$\text{mean}(\vec{x_1}, \vec{x_2}, \dots, \vec{x_n}) = \frac{1}{N} \sum_i \vec{x_i}$$

The mean vector is the **geometric centroid** of a set of vectors and can be thought of as capturing "centre of mass" of those vectors. 

If we have vectors stacked up in a matrix $X$, one vector per row, `np.mean(x, axis=0)` will calculate the mean vector for us:

We can **center** a dataset stored as an array of vectors to **zero mean** by just subtracting the mean vector from every row.

### Median is harder

Note that other statistical operations like the median can be generalised to higher dimensions, but it is much more complex to do so, and there is no simple direct algorithm for computing the **geometric median**. This is because the are *not* just combined operations of scalar multiplication and vector addition.

### High\-dimensional vector spaces

Vectors in low dimensional space, such as 2D and 3D are familiar in their operation. However, data science often involves **high dimensional vector spaces**, which obey the same mathematical rules as we have defined, but whose properties are sometimes unintuitive.

Many problems in machine learning, optimisation and statistical modelling involve using *many measurements*, each of which has a simple nature; for example, an image is just an array of luminance measurements. A 512x512 image could be considered a single vector of 262144 elements. We can consider one "data point" to be a vector of measurements. The dimension $d$ of these "feature vectors" has a massive impact on the performance and behaviour of algorithmics and many of the problems in modelling are concerned with dealing with high\-dimensional spaces.

High-dimensional can mean any $d>3$; a 20-dimensional feature set might be called medium\-dimensional; a 1000\-dimensional might be called high-dimensional; a 1M\-dimensional dataset might be called extremely high\-dimensional. These are loose terms, and vary from discipline to discipline.

#### Geometry in high\-D
The geometric properties of high\-d spaces are very counter\-intuitive. The volume of space increases exponentially with $d$ \(e.g. the volume of a hypersphere or hypercube\). There is *a lot* of empty space in high\-dimensions, and where data is sparse it can be difficult to generalise in high-dimensional spaces. Some research areas, such as genetic analysis often have i.e. many fewer samples than measurement features \(we might have 20000 vectors, each with 1 million dimensions\). 
 
### Curse of dimensionality    

Many algorithms that work really well in low dimensions break down in higher dimensions. This problem is universal in data science and is called the **curse of dimensionality**. Understanding the curse of dimensionality is critical to doing any kind of data science.

#### High\-D histograms don't work

If we had 10 different measurements \(air temperature, air humidity, latitude, longitude, wind speed, wind direction, precipitation, time of day, solar power, sea temperature\) and we wanted to subdivide them into 20 bins each, we would need a histogram with $20^{10}$ bins -- over ***10 trillion*** bins. 

But we only have 10,000 measurements; so we'd expect that virtually every bin would be empty, and that a tiny fraction of bins \(about 1 in a billion in this case\) would have probably one measurement each. Not to mention a naive implementation would need memory space for 10 trillion counts -- even using 8 bit unsigned integers this would be 10TB of memory!

This is the problem of sparseness in high\-dimensions. There is a lot of volume in high\-D, and geometry does not work as you might expect generalising from 2D or 3D problems.

* **Curse of dimensionality: as dimension increases generalisation gets harder *exponentially***

### Paradoxes of high dimensional vector spaces

Here are some high\-d "paradoxes":

#### Think of a box

* Imagine an empty box in high\-D \(a hyper cube\). 
    * Fill it with random points. For any given point, in high enough dimension, the boundaries of the box will be closer than any other point in the box.
    * Not only that, but every point will be nearly the same \(Euclidean, $L_2$\) distance away from any other point.
    * The box will have $2^d$ corners. For example, a 20D box has more than 1 million corners.
    * For $d>5$ more of the volume is in the areas close to the corners than anywhere else; by $d=20$  the overwhelming volume of the space is in the corners.
    * Imagine a sphere sitting in the box so the sphere's surface just touches the edges of the box \(an inscribed sphere\). As D increases, the sphere takes up less and less of the box, until it is a vanishingly small proportion of the space.
    * Fill the inner sphere with random points. **Almost all of them** are within in a tiny shell near the surface of the sphere, with virtually none in the centre.

#### Spheres in boxes

Although humans are terrible at visualising high dimensional problems, we can see some of the properties of high\-d spaces visually by analogy.

#### Lines between points

Even if we take two random points in a high\-dimensional cube, and then draw a line between those points *the points on the line still end up on the edge of the space*! There's no way in.

#### Distances don't work so well

If we compute the distance between two random high\-dimensional vectors in the Euclidean norm, the results will be *almost the same*. Almost all points will be a very similar distance apart from each other. 

Other norms like the $L_{\inf}$ or $L_1$ norm, or the cosine distance \(normalised dot product\) between two vectors $\vec{x}$ and $\vec{y}$ are less sensitive to high dimensional spaces, though still not perfect.

### Matrices and linear operators

#### Uses of matrices

We have seen that \(real\) vectors represent elements of a vector space as 1D arrays of real numbers \(and implemented as ndarrays of floats\). 

Matrices represent **linear maps** as 2D arrays of reals; $\real^{m\times n}$.

* Vectors represent "points in space"
* Matrices represent *operations* that do things to those points in space. 

The operations represented by matrices are a particular class of functions on vectors -- "rigid" transformations. Matrices are a very compact way of writing down these operations.

#### Operations with matrices

There are many things we can do with matrices:

* They can be added and subtracted $C=A+B$ 
    *  $(\real^{n\times m},\real^{n\times m}) \rightarrow \real^{n\times m}$
* They can be scaled with a scalar $C = sA$
    * $(\real^{n\times m},\real) \rightarrow \real^{n\times m}$
* They can be transposed $B = A^T$; this exchanges rows and columns
    * $\real^{n\times m} \rightarrow \real^{m\times n}$
* They can be *applied to vectors* $\vec{y} = A\vec{x}$; this **applies** a matrix to a vector.
    * $(\real^{n\times m}, \real^{m}) \rightarrow \real^{n}$
* They can be *multiplied together* $C = AB$; this **composes** the effect of two matrices 
    * $(\real^{p\times q}, \real^{q\times r})\rightarrow \real^{p\times r}$

#### Intro to matrix notation

We write matrices as a capital letter: 

$$A \in \real^{n \times m}=  \begin{bmatrix}
a_{1,1}  & a_{1,2} & \dots & a_{1,m}  \\
a_{2,1}  & a_{2,2}  & \dots & a_{2,m}  \\
\dots \\
a_{n,1} + & a_{n,2}  & \dots & a_{n,m} \\
\end{bmatrix},\  a_{i,j}\in \real$$

\(although we don't usually write matrices with capital letters in code -- they follow the normal rules for variable naming like any other value\)

A matrix with dimension $n \times m$ has $n$ rows and $m$ columns \(remember this order -- it is important!\). Each element of the matrix $A$ is written as $a_{i,j}$ for the $i$th row and $j$th column.

Matrices correspond to the 2D arrays / rank-2 tensors we are familiar with from earlier. But they have a very rich mathematical structure which makes them of key importance in computational methods. *Remember to distinguish 2D arrays from the mathematical concept of matrices. Matrices \(in the linear algebra sense\) are represented by 2D arrays, as real numbers are represented by floating\-point numbers*

### Geometric intuition \(cube \-\> parallelepiped\)

An intuitive way of understanding matrix operations is to consider a matrix to transform a cube of vector space centered on the origin in one space to a **parallelotope** in another space, with the origin staying fixed. This is the *only* kind of transform a matrix can apply.

A parallelotope is the generalisation of a parallelogram to any finite dimensional vector space, which has parallel faces but edges which might not be at 90 degrees.

#### Transforms and projections

* A linear map is any function $f$ $R^m \rightarrow R^n$ which satisfies the linearity requirements.
* If the map represented by the matrix is $n\times n$ then it maps from a vector space onto the *same* vector space \(e.g. from $\real^n \rightarrow \real^n$\), and it is called a **linear transform**.
* If the map has the property $Ax = AAx$ or equivalently $f(x)= f(f(x))$ then the operation is called a **linear projection**; for example, projecting 3D points onto a plane; applying this transform to a set of vectors twice is the same as applying it once.

### Keeping it real

We will only consider **real matrices** in this course, although the abstract definitions above apply to linear maps across any vector space \(e.g complex numbers, finite fields, polynomials\).

#### Linear maps are representable as matrices

*Every linear map of real vectors can be written as a real matrix.* In other words, if there is a function $f(\vec{x})$ that satisfies the linearity conditions above, it can be expressed as a matrix $A$.

### Matrix operations

There is an **algebra** of matrices; this is **linear algebra**. In particular, there is a concept of addition of matrices of *equal size*, which is simple elementwise addition:


<div class="alert alert-box alert-success">
    
$$  A + B = \begin{bmatrix}
a_{1,1} + b_{1,1} & a_{1,2} + b_{1,2} & \dots & a_{1,m} + b_{1,m} \\
a_{2,1} + b_{2,1} & a_{2,2} + b_{2,2} & \dots & a_{2,m} + b_{2,m} \\
\dots \\
a_{n,1} + b_{n,1} & a_{n,2} + b_{n,2} & \dots & a_{n,m} + b_{n,m} \\
\end{bmatrix}
$$
</div>

along with scalar multiplication $cA$, which multiplies each element by $c$.


<div class="alert alert-box alert-success">
 
$$  cA = \begin{bmatrix}
ca_{1,1}  & ca_{1,2} & \dots & ca_{1,m}  \\
ca_{2,1} & ca_{2,2}  & \dots & ca_{2,m} \\
\dots \\
ca_{n,1}  & ca_{n,2} & \dots & ca_{n,m} \\
\end{bmatrix}
$$
</div>

These correspond exactly to addition and scalar multiplication in NumPy.

### Application to vectors

We can apply a matrix to a vector. We write it as a product $A\vec{x}$, to mean the matrix $A$ applied to the vector $\vec{x}$.  This is equivalent to applying the function $f(\vec{x})$, where $f$ is the corresponding function.

If $A$ is $\real^{n \times m}$, and $\vec{x}$ is $\real^m$, then this will map from an $m$ dimensional vector space to an $n$ dimensional vector space.

**All application of a matrix to a vector does is form a weighted sum of the elements of the vector**. This is a linear combination \(equivalent to a "weighted sum"\) of the components.

In particular, we take each element of $\vec{x}, x_1, x_2, \dots, x_m$, multiply it with the corresponding *column* of $A$, and sum these columns together.

### Matrix multiplication

Multiplication is the interesting matrix operation. Matrix multiplication defines the product $C=AB$, where $A,B,C$ are all matrices.

Matrix multiplication is defined such that if $A$ represents linear transform $f(\vec{x})$ and
$B$ represents linear transform $g(\vec{x})$, then $BA\vec{x} = g(f(\vec{x}))$.

**Multiplying two matrices is equivalent to composing the linear functions they represent, and it results in a matrix which has that affect.**

*Note that the composition of linear maps is read right to left. To apply the transformation $A$, **then** $B$, we form the product $BA$, and so on.*

### Multiplication algorithm

This gives rise to many important uses of matrices: for example, the product of a scaling matrix and a rotation matrix is a scale-and-rotate matrix. It also places some requirements on the matrices which form a valid product. Multiplication is *only* defined for two matrices $A, B$ if:
* $A$ is $p \times q$ and
* $B$ is $q \times r$.

This follows from the definition of multiplication: $A$ represents a map $\real^q \rightarrow \real^p$ and $B$ represents a map $\real^r \rightarrow \real^q$. The output of $A$ must match the dimension of the input of $B$, or the operation is undefined. 

Matrix multiplication is defined in a slightly surprising way, which is easiest to see in the form of an algorithm:
    
### Time complexity of multiplication

Matrix multiplication has, in the general case, of time complexity $O(pqr)$, or for multiplying two square matrices $O(n^3)$. This is apparent from the three nested loops above. However, there are many special forms of matrices for which this complexity can be reduced, such as diagonal, triangular, sparse and banded matrices. We will we see these **special forms** later.

There are some accelerated algorithms for general multiplication. The time complexity of all of them is $>O(N^2)$ but $<O(N^3)$. Most accelerated algorithms are impractical for all but the largest matrices because they have enormous constant overhead.

### Transposition

The **transpose** of a matrix $A$ is written $A^T$ and has the same elements, but with the rows and columns exchanged. Many matrix algorithms use transpose in computations.

### Composed maps 

There is a very important property of matrices. If $A$ represents $f(x)$ and $B$ represents $g(x)$, then the product $BA$ represents $g(f(x))$. **Multiplication is composition.** Note carefully the order of operations. $BA\vec{x} = B(A\vec{x})$ means do $A$ to $\vec{x}$, then do $B$ to the result.

We can visually verify that composition of matrices by multiplication is the composition of their effects. 

### Concatenation of transforms

Many software operations take advantage of the definition of matrix multiplication as the composition of linear maps. In a graphics processing pipeline, for example, all of the operations to position, scale and orient visible objects are represented as matrix transforms. Multiple operations can be combined into *one single matrix operation*.

The desktop UI environment you are uses linear transforms to represent the transformation from data coordinates to screen coordinates. Because multiplication composes transforms, only a single matrix for each object needs to be kept around. \(actually for 3D graphics, at least two matrices are kept: one to map 3D \-\> 3D \(the *modelview matrix*\) and one to map 3D \-\> 2D \(the *projection matrix*\)\).

Rotating an object by 90 degrees computes product of the current view matrix with a 90 degree rotation matrix, which is then stored in place of the previous view matrix. This means that all rendering just needs to apply to relevant matrix to the geometry data to get the pixel coordinates to perform the rendering.

### Commutativity

The order of multiplication is important. Matrix multiplication does **not** commute; in general:

$$AB \neq BA$$

This should be obvious from the fact that multiplication is only defined for matrices of dimension $p \times q$ and $q \times r$; unless $p=q=r$ then the multiplication is not even *defined* if the operands are switched, since it would involve a $q \times r$ matrix by a $p \times q$ one!

Even if two matrices are compatible in dimension when permuted \(i.e. if they are square matrices, so $p=q=r$\), multiplication still does not generally commute and it matters which order operations are applied in.

#### Transpose order switching

There is a very important identity which is used frequently in rearranging expressions to make computation feasible. That is:

$$(AB)^T = B^TA^T$$
    
Remember that matrix multiplication doesn't commute, so $AB \neq BA$ in general \(though it can be true in some special cases\), so this is the only way to algebraically reorder general matrix multiplication expressions \(side note: inversion has the same effect, but only works on non-singular matrices\). This lets us rearrange the order of matrix multiplies to "put matrices in the right place".

It is also true that $$(A+B)^T = A^T+B^T$$ but this is less often useful.

### Left\-multiply and right\-multiply

Because of the noncommutativity of multiplication of matrices, there are actually two different matrix multiplication operations: **left multiplication** and **right multiplication**.

$B$ left-multiply $A$ is $AB$; $B$ right-multiply $A$ is $BA$. This becomes important if we have to multiply out a longer expression:

$$B\vec{x}+\vec{y}\\
\text{left multiply by A}\quad =A[B\vec{x}+\vec{y}] = AB\vec{x} + A\vec{y}\\
\text{right multiply by A}\quad =[B\vec{x}+\vec{y}]A = B\vec{x}A + \vec{y}A\\
$$

### Covariance ellipses

This matrix captures the spread of data, including any **correlations** between dimensions. It can be seen as capturing an **ellipse** that represents a dataset.  The **mean vector** represents the centre of the ellipse, and the **covariance matrix** represent the shape of the ellipse. This ellipse is often called the **error ellipse** and is a very useful summary of high-dimensional data.

The covariance matrix represents a \(inverse\) transform of a unit sphere to an ellipse covering the data. Sphere->ellipse is equivalent to square\-\>parallelotope and so can be precisely represented as a matrix transform.

### Anatomy of a matrix

The **diagonal** entries of a matrix ($A_{ii}$) are important "landmarks" in the structure of a matrix. Matrix elements are often referred to as being "diagonal" or "off\-diagonal" terms. 

### Zero

The zero matrix is all zeros, and is defined for any matrix size $m\times n$. It is written as $0$. Multiplying any vector or matrix by the zero matrix results in a result consisting of all zeros. The 0 matrix maps all vectors onto the zero vector \(the origin\).

### Square

A matrix is square if it has size $n\times n$. Square matrices are important, because they apply transformations *within* a vector space; a mapping from $n$ dimensional space to $n$ dimensional space; a map from $\real^n \rightarrow \real^n$. 

They represent functions mapping one domain to itself. Square matrices are the only ones that:
* have an inverse 
* have determinants
* have an eigendecomposition

which are all ideas we will see in the following unit.

### Triangular

A square matrix is triangular if it has non-zero elements only above \(**upper triangular**\) or below the diagonal \(**lower triangular**\), *inclusive of the diagonal*.

**Upper triangular**
$$\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & 5 & 6 & 7 \\
0 & 0 & 8 & 9 \\
0 & 0 & 0 & 10 \\
\end{bmatrix}
$$


**Lower triangular**
$$\begin{bmatrix}
1 & 0 & 0 & 0 \\
2 & 3 & 0 & 0 \\
4 & 5 & 6 & 0 \\
7 & 8 & 9 & 10 \\
\end{bmatrix}
$$


These represent particularly simple to solve sets of simultaneous equations. For example, the lower triangular matrix above can be seen as the system of equations:

$$x_1 = y_1\\
2x_1 + 3x_2 = y_2\\ 
4x_1 + 5x_2 + 6x_3 = y_3\\ 
7x_1 + 8x_2 + 9x_3 + 10x_4 = y_4\\ 
$$

which, for a given $y_1, y_2, y_3, y_4$ is trivial to solve by substitution.

