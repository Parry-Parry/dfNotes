
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>

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

_We need to explicitly copy arrays if we want to work on a new array<sub>


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
          z = zero_array<sub>like(x);
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

If we have a triplet of vector $\vec{x},\vec{y<sub>2},\vec{y<sub>1}$, where both $y$s match the $x$ then the area between the two lines can be drawn using polygon **geoms**. This results in a **ribbon plot**, where the ribbon can have variable thickness \(as the difference $y<sub>1-y<sub>2$ varies\).

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

Plots of a single 1D array of numbers $[x<sub>1, x<sub>2, \dots, x<sub>n]$ usually involve displaying *statistics* \(**stats**\)  to transform the dataset into forms that 2D plotting forms can be applied to. Very often these plot types are used with multiple arrays \(data that has been grouped in some way\) to show differences between the groups.

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

Cartesian plots are the most familiar coords. But some data is naturally represented in different mappings. The most notable of these is **polar coordinates**, in which two values are mapped onto an *angle* $θ$ and a *radius* $r$. 

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
In this course, we will consider vectors to be ordered tuples of real numbers $[x<sub>1, x<sub>2, \dots x<sub>n], x<sub>i \in \mathbb{R}$ (the concept generalises to complex numbers, finite fields, etc. but we'll ignore that). A vector has a fixed dimension $n$, which is the length of the tuple. We can imagine each element of the vector as representing a distance in an **direction orthogonal** to all the other elements.

For example, a length-3 vector might be used to represent a spatial position in Cartesian coordinates, with three orthogonal measurements for each vector. Orthogonal just means "independent", or, geometrically speaking "at 90 degrees".


* Consider the 3D vector [5, 7, 3]. This is a point in $\real^3$, which is formed of:

            5 * [1,0,0] +
            7 * [0,1,0] +
            3 * [0,0,1]
            
Each of these vectors [1,0,0], [0,1,0], [0,0,1] is pointing in a independent direction (orthogonal direction) and has length one. The vector [5,7,3]
can be thought of a weighted sum of these orthogonal unit vectors (called **"basis vectors"**). The vector space has three independent bases, and so is three dimensional.

We write vectors with a bold lower case letter:
$$\vec{x} = [x<sub>1, x<sub>2, \dots, x<sub>d],\\
\vec{y} = [y<sub>1, y<sub>2, \dots, y<sub>d],$$ and so on.

#### Points in space

##### Notation: $\real^n$

* $\real$ means the set of real numbers.  
* $\real_{\geq 0}$ means the set of non-negative reals.  
* $\real^n$ means the set of tuples of exactly $n$ real numbers. 
* $\real^{n\times m}$ means the set of 2D arrays (matrix) of real numbers with exactly $n$ rows of $m$ elements.

* The notation $(\real^n, \real^n) \rightarrow \real$ says that than operation defines a map from a pair of $n$ dimensional vectors to a real number.

##### Vector spaces

Any vector of given dimension $n$ lies in a **vector space**, called $\real^n$ (we will only deal with finite-dimensional real vector spaces with standard bases), which is the set of possible vectors of length $n$ having real elements, along with the operations of:   
*  **scalar multiplication** so that $a{\bf x}$  is defined for any scalar $a$. For real vectors, $a{\bf x} = [a x<sub>1, a x<sub>2, \dots a x<sub>n]$, elementwise scaling.
    * $(\real, \real^n) \rightarrow \real^n$
* **vector addition** so that ${\bf x} + {\bf y}$ vectors ${\bf x, y}$ of equal dimension. For real vectors, ${\bf x} + {\bf y} = [x<sub>1 + y<sub>1, x<sub>2 + y<sub>2, \dots x<sub>d + y<sub>d]$ the elementwise sum
    * $(\real^n, \real^n) \rightarrow \real^n$


We will consider vector spaces which are equipped with two additional operations:  
* a **norm** $||{\bf x}||$ which allows the length of vectors to be measured.
    * $\real_n \rightarrow \real_{\geq 0}$
* an **inner product** $\langle {\bf x} | {\bf y} \rangle$ or ${\bf x \bullet y}$  which allows the angles of two vectors to be compared. The inner product of two orthogonal vectors is 0. For real vectors ${\bf x}\bullet{\bf y} = x<sub>1 y<sub>1 + x<sub>2 y<sub>2 + x<sub>3 y<sub>3 \dots x<sub>d y<sub>d$
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

One of the simplest effective machine learning algorithms is **k nearest neighbours**. This involves some *training set* of data, which consists of pairs $\vec{x<sub>i}, y<sub>i$: a feature vector $\vec{x<sub>i}$ and a label $y<sub>i$. 

When a new feature needs classified to make a prediction, the $k$ *nearest* vectors in this training set are computed, using a **norm** to compute distances. The output prediction is the class label that occurs most times among these $k$ neighbours \($k$ is preset in some way; for many problems it might be around 3\-12\).

The idea is simple; nearby vectors ought to share common properties. So to find a property we don't know for a vector we do know, look at the properties that nearby vectors have.

### Image compression

Images have a straightforward representation as 2D arrays of brightness, as we have seen already. But, just like text, this representation is rather empty in terms of the operations that can be done to it. A single pixel, on its own, has as little meaning as a single letter.

Groups of pixels -- for example, rectangular patches -- can be unraveled into a vector. An 8x8 image patch would be unraveled to a 64\-dimensional vector. These vectors can be treated as elements of a vector space.

Many image compression algorithms take advantage of this view. One common approach involves splitting images into patches, and treating each patch as a vector $\vec{x<sub>1}, \dots, \vec{x<sub>n}$. The vectors are **clustered** to find a small number of vectors $\vec{y<sub>1}, \dots, \vec{y<sub>m},\ m << n$ that are a reasonable approximation of nearby vectors. Instead of storing the whole image, the vectors for the small number of representative vectors $\vec{y<sub>i}$ are stored \(the **codebook**\), and the rest of the image is represented as the *indices* of the "closest" matching vector in the codebook i.e. the vector $\vec{y<sub>j}$ that minimises $||x<sub>i - y<sub>j||$. 

This is **vector quantisation**, so called because it quantises the vector space into a small number of discrete regions. This process maps **visual similarity onto spatial relationships.**

### Basic vector operations

There are several standard operations defined for vectors, including getting the length of vectors,  and computing dot (inner), outer and cross products.

#### Addition and multiplication

Elementwise addition and scalar multiplication on arrays already implement the mathematical vector operations. Note that these ideas let us form **weighted sums** of vectors:  
$$\lambda_1 \vec{x<sub>1} + \lambda_2 \vec{x<sub>2} + \dots + \lambda_n \vec {x<sub>n}$$

This applies **only** to vectors of the same dimension.

### How big is that vector?

Vector spaces do not necessarily have a concept of distance, but the spaces we will consider can have a distance *defined*. It is not an inherent property of the space, but something that we define such that it gives us useful measures.

The Euclidean length of a vector $\bf x$ (written as $||{\bf x}||$) can be computed directly using `np.linalg.norm()`. This is equal to:

$$ \|{\bf x}\|\_2 = \sqrt{x<sub>0^2 + x<sub>1^2 + x<sub>2^2 + \dots + x<sub>n^2  } $$

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

$$\text{mean}(\vec{x<sub>1}, \vec{x<sub>2}, \dots, \vec{x<sub>n}) = \frac{1}{N} ∑ \vec{x<sub>i}$$

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

In particular, we take each element of $\vec{x}, x<sub>1, x<sub>2, \dots, x<sub>m$, multiply it with the corresponding *column* of $A$, and sum these columns together.

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

$$x<sub>1 = y<sub>1\\
2x<sub>1 + 3x<sub>2 = y<sub>2\\ 
4x<sub>1 + 5x<sub>2 + 6x<sub>3 = y<sub>3\\ 
7x<sub>1 + 8x<sub>2 + 9x<sub>3 + 10x<sub>4 = y<sub>4\\ 
$$

which, for a given $y<sub>1, y<sub>2, y<sub>3, y<sub>4$ is trivial to solve by substitution.

## Week 5

### Graphs as matrices

We might model the connectivity of distribution centres as a graph. A directed graph connects vertices by edges. The definition of a graph is  
G = \(V, E\), where V is a set of vertices and E is a set of edges connecting pairs of vertices.

### Computing graph properties

There are some graph properties which we can compute easily from this binary matrix:  
* The *out-degree* of each vertex \(number of edges leaving a vertex\) is the sum of each row.
* The *in-degree* of each vertex \(number of edges entering a vertex\) is the sum of each column.
* If the matrix is symmetric it represents an undirected graph; this is the case if it is equal to its transpose.
* A directed graph can be converted to an undirected graph by computing $A^\prime = A + A^T$. This is equivalent to making all the arrows bi\-directional.
* If there are non\-zero elements on the diagonal, that means there are edges connecting vertices to themselves \(self\-transitions\).

### Edge\-weighted graphs

If the some of the connections between distibution centres are stronger than others, e.g. if they are connected by bigger roads, we can model this using edge weights. Now the entry at A<sub>ij</sub> represents the weight of the connection from vertex V<sub>i</sub> to V<sub>j</sub>.

We can think of graphs as representing flows of "mass" through a network of vertices.

* If the total flow out of a vertex is \>1, i.e. its row sums to \>1, then it is a **source** of mass; for example it is *manufacturing* things.
* If the total flow out of a vertex is <1, i.e. its row sums to <1, then it is a **sink**; for example it is *consuming* things.
* If the total flow out of the vertex is 1 exactly, i.e. its row sums to 1 exactly, then it conserves mass; it only ever *re\-routes* things.

If the whole graph consists of vertices whose total outgoing weight is 1.0, and all weights are positive or zero, then the whole graph preserves mass under flow. Nothing is produced or consumed. Every row in the adjacency matrix $A$ sums to 1. This is called a **conserving adjacency matrix**. We can normalise the rows of any positive matrix A \(so long as each vertex has at least some flow out of it\) to form a conserving adjacency matrix.

### Flow analysis: using matrices to model discrete problems

Previously, we have talked about how matrices perform geometrical transformations on vectors. Matrices are sometimes referred to as linear maps because they map from one vector to another, using only linear operations. The adjacency matrix does this too.

At any point in time, we can write down the proportion of packages at each depot as a vector →x<sub>t</sub> ∈ V, where V is the number of vertices in the graph \(number of depots\).

The flow of packages \(per day\) between depots is a linear map V → V. This is represented by the adjacency matrix A ∈ V×V \(a square matrix\).

The switch of viewpoints from discrete to continuous \(and vice versa\) is a very powerful and fundamental step in data analysis. It might not seem like it, but the flow of packages can be modelled as some rigid rotation and scaling in a high\-dimensional space.

A great many problems can be represented this way:

* packages moving between depots \(how many packages at each depot at an instant in time\)
* users moving between web pages \(how many users on each webpage\)
* cancer cells moving between tumour sites \(how many cancer cells at each tumour site\)
* trade between states \(how many items in each state\)
* shoppers walking between retailers \(how many shoppers in each shop\)
* traffic across a load-balancing network \(how many cars at each junction\)
* NPCs (non-player characters) moving between towns in a game \(how many NPCs in each town\)
* fluid moving between regions \(how much fluid in each tank\)
* blood flowing between organs \(how much blood in each organ\)
* beliefs moving among hypotheses \(how much we believe in each hypothesis\)

### Simulation of changes in package distribution over time

Suppose we start with an initial distribution of packages: a vector x<sup>→</sup><sub>t</sub>=0. How many packages will be at each depot tomorrow? This will be given by the vector x<sup>→</sup><sub>t</sub>=1, which can be computed as follows:  
x<sup>→</sup><sub>t=1</sub> = x<sup>→</sup><sub>t=0</sub>A  

We can simulate the flow *over the whole network* in one go with just one matrix multiplication. This "rotates" the distribution of packages from today to tomorrow. The advantage of vectorised operations is that they can be accelerated using hardware such as a GPU \(Graphics Processing Unit\).

### Some new matrix operations

For square matrices A \(i.e. representing **linear transforms**\):  
* Matrices can be exponentiated: C = A^n this "repeats" the effect of matrix 
* Matrices can be inverted: C = A^\{-1\} this undoes the effect of a matrix 
* We can find eigenvalues: A\{x\} = λ \{x\} this identifies specific vectors \{x\} that are *only* scaled by a factor λ \(not rotated\) when transformed by matrix A.
* Matrices can be factorised: A = UΣV<sup>T</sup> any matrix can expressed as the product of three other matrices with special forms. 
* We can measure some properties of A numerically, including the **determinant**, **trace** and **condition number**.

### Eigenvalues and eigenvectors

A matrix represents a special kind of function: a **linear transform**; an operation that performs rotation and scaling on vectors. However, there are certain vectors which don't get rotated when multiplied by the matrix. They only get scaled \(stretched or compressed\). These vectors are called **eigenvectors**, and they can be thought of as the "fundamental" or "characteristic" vectors of the matrix, as they have some stability. The prefix **eigen** just means **characteristic**. The scaling factors that the matrix applies to its eigenvectors are called **eigenvalues**.

We can visualise the effect of a matrix transformation by imagining a parallelepiped \(whose edges are vectors\) being rotated, stretched and compressed. If the edges of the parallelelpiped are the eigenvectors of the matrix, the parallelepiped will only be stretched or compressed, not rotated. If the edges of this parallelepiped have unit length, then after the transformation their lengths will be equal to the eigenvalues. 

#### How to find the leading eigenvector: the power iteration method

What happens if we apply a square matrix A to a vector x of any length, then take the resulting vector and apply A again, and so on?

Let's work with column vectors now, so that A pre\-multiplies the vector. If we compute

x<sub>n</sub> = A<sup>n</sup>x<sub>0</sub>

this will generally either explode in value or collapse to zero. However, we can fix the problem by normalizing the resulting vector after each application of the matrix.

The vector will be forced back to unit norm at each step, using the L<sub>∞</sub> norm. This process is called power iteration.

The vector that results from power iteration is known as the leading eigenvector. It satisfies the definition of an eigenvector because the matrix A performs only scaling on this vector \(no rotation\). We know this **must** be true, because the scaling effect is eliminated by the normalisation step in the power iteration, but any other effects pass through.

### Formal definition of eigenvectors and eigenvalues

Consider a vector function f\(x\). There may exist vectors such that f\(x\) = λx. The function maps these vectors to scaled versions of themselves. No rotation or skewing is applied, just pure scaling.

Any square matrix A represents a function f\(x\) and may have vectors like this, such that

Ax<sub>i</sub> = λ<sub>i</sub>x<sub>i</sub>

Each vector xi satisfying this equation is known as an **eigenvector** and each corresponding factor λ<sub>i</sub> is known as an eigenvalue.

For any matrix, the eigenvalues are uniquely determined, but the eigenvectors are not. There may be many eigenvectors corresponding to any given eigenvalue.

Eigenproblems are problems that can be tackled using eigenvalues and eigenvectors.

From the **eigendecomposition** we can get a feel for which eigenvectors are large and which are small.

### The eigenspectrum

The **eigenspectrum** is just the sequence of absolute eigenvalues, ordered by magnitude λ<sub>1</sub>\>λ<sub>2</sub>\>...\>λ<sub>n</sub>. This *ranks* the eigenvectors in order of "importance". As we shall see later, this can be useful in finding "simplified" versions of linear transforms.

### Numerical instability of eigendecomposition algorithms

A word of warning: `np.linalg.eig` can suffer from numerical instabilities due to rounding errors resulting from limitations on floating point precision. This means that sometimes the smallest eigenvectors are not completely orthogonal. `np.linalg.eig` is often sufficient for most purposes, but be careful how you use it. 

If your matrix satisfies certain special conditions, you might be able to use a more stable algorithm. For example, if it is real and symmetric \(or Hermitian, in the case of a complex matrix\), you can use `np.linalg.eigh`. 

### Principal Component Analysis \(PCA\)

We saw the covariance matrix Σ in the previous Unit. It tells us how much correlation there is between the variables in a data set. We can plot a representation of the covariance matrix as an ellipse which aligns with the distribution of the data points. Uncorrelated data is represented by a circle, whereas strongly correlated data is represented by a long thin ellipse. The eigenvectors of the covariance matrix, scaled by their eigenvalues, form the principal axes of the ellipse.

#### Decomposition of the covariance matrix into its eigenvectors and eigenvalues

The eigenvectors of the covariance matrix are called the **principal components**, and they tell us the directions in which the data varies most. This is an incredibly useful thing to be able to do, particularly with high\-dimensional data sets where the variables may be correlated in complicated ways.

The direction of principal component i is given by the eigenvector x<sub>i</sub>, and the length of the component is given by sqrt\(λ<sub>i</sub>\).


### Reconstruction of the covariance matrix from its eigenvectors and eigenvalues

Since we are able to decompose the covariance matrix into its constituent eigenvectors and eigenvalues, we must also be able to use these constituent parts to reconstruct the covariance matrix. We can do this as follows:

Σ = Q λ Q<sup>T</sup>

where Q is a matrix of unit eigenvectors x<sub>i</sub> (same as the output `np.linalg.eig`) and λ is a diagonal matrix of eigenvalues \(λ<sub>i</sub> on the diagonal, zero elsewhere\).

### Approximating a matrix

Imagine we started with a very high dimensional data set, so $σ$ is a very large matrix. It's so large, we don't want to store it in memory. Instead, we just want to store the first few principal components and use these to reconstruct an *approximation* to Σ. Providing we keep the largest principal components, we will probably retain most of the information. 


Matrix approximation can be used to simplify transformations or to compress matrices for data transmission.

The eigenspectrum gives us an idea of how simply a matrix could be approximated:
* One large eigenvalue and many small ones \- just one vector might approximate this matrix. 
* All eigenvalues similar magnitude? We will not be able to approximate this transform easily.


### Dimensionality reduction

We can also reduce the dimensionality of our original dataset by projecting it onto the few principal components of the covariance matrix that we've kept. We can do this by multiplying the dataset matrix by each component and saving the projected data into a new, lower\-dimensional matrix.


### Uses of eigendecomposition

Matrix decomposition is an *essential* tool in data analysis. It can be extremely powerful and is efficient to implement. Systems such as recommenders \(e.g. Netflix, YouTube, Amazon, etc.\), search engines \(Google\), image compression algorithms, machine learning tools and visualisation systems apply these decompositions *extensively*.

Google was wholly built around matrix decomposition algorithms; that's what "PageRank" is. This is what allowed Google to race ahead of their competitors in the early days of the search wars.

The eigendecomposition can be used *anywhere* there is a system modelled as a linear transform \(any linear map N → N\). It lets us predict behaviour over different time scales \(e.g. very short term or very long term\). For instance, we can:

* Find "modes" or "resonances" in a system \(e.g physical model of a bridge\). 
    * For example, every room has a set of "eigenfrequencies" - the acoustic resonant modes. A linear model of the acoustics of the room could be written as a matrix, and the resonant frequencies extracted directly via the eigendecomposition. [
* Predict the behaviour of feedback control systems: is the autopilot going to be stable or unstable?
* Partition graphs and cluster data \(spectral clustering)\. 
* Predict flows on graphs.
* Perform Principal Component Analysis on high\-dimensional data sets for exploratory data analysis, 2D visualisation or data compression.

As soon as we can write down a matrix A we can investigate its properties with the eigendecomposition.

### Matrix properties from the eigendecomposition

### Trace

The trace of a square matrix can be computed from the sum of its diagonal values:

Tr\(A\) = a<sub>1,1</sub> + a<sub>2,2</sub> \+ ... \+ a<sub>n,n</sub>

It is also equal to the sum of the eigenvalues of A

Tr\(A\) = sum\(λ<sub>i</sub>\) for i in range 1 to n

The trace can be thought of as measuring the  **perimeter** of the parallelotope of a unit cube transformed by the matrix.

### Determinant

The determinant det\(A\) is an important property of square matrices. It can be thought of as the **volume** of the parallelotope  of a unit cube transformed by the matrix \-\- it measures how much the space expands or contracts after the linear transform.

It is equal to the product of the eigenvalues of the matrix.

det\(A\) = product\(λ<sub>i</sub>\) 

If any eigenvalue λ<sub>i</sub> of A is 0, the determinant det\(A\)=0, and the transformation collapses at least one dimension to be completely flat. This means that the transformation **cannot be reversed**; information has been lost.

### Definite and semi-definite matrices

A matrix is called **positive definite** if all of its eigenvalues are greater than zero: λ<sub>i</sub> \> 0$. It is **positive semi\-definite** if all of its eigenvalues are greater than or equal to zero: λ<sub>i</sub> ≥ 0. It is **negative definite** if all of the eigenvalues are less than zero: λ<sub>i</sub> < 0, and **negative semi\-definite** if all the eigenvalues are less than or equal to zero: λ<sub>i</sub> ≤ 0.

A positive definite matrix A has the property x<sup>T</sup>Ax\>0 for all \(nonzero\) x. This tells us that the dot product of x with Ax must be positive \(N.B. Ax is the vector obtained by transforming x with A\). This can only happen if the angle θ between x and Ax is less than π\/2, since x<sup>T</sup>Ax = \|x\|\|Ax\|cos\(θ\). That means that A does not rotate x through more than 90∘.

Positive definiteness will be an important concept when we discuss **covariance matrices** \(important in statistical data analysis\) and **Hessian matrices**  \(important in numerical optimisation\).

### Matrix Inversion

We have seen four basic algebraic operations on matrices:

* scalar multiplication cA
* matrix addition A\+B
* matrix multiplication BA
* matrix transposition A<sup>T</sup>

There is a further important operation: **inversion** A<sup>-1</sup>, defined such that:

* A<sup>-1</sup>(Ax) = x
* A<sup>-1</sup>A = I
* \(A<sup>-1</sup>\)<sup>-1</sup> = A
* \(AB\)<sup>-1</sup> = B<sup>-1</sup>A<sup>-1</sup>

The equivalent of division for matrices is left\-multiplication by the inverse. This has the effect reversing or undoing the effect of the original matrix. 

*Inversion is only defined for certain kinds of matrices, as we will see below.*

### Only square matrices can be inverted

Inversion is only defined for square matrices, representing a linear transform $\real^n \rightarrow \real^n$. This is equivalent to saying that the determinant of the matrix must be non\-zero: $\det(A) \neq 0$. Why?

A matrix which is non-square maps vectors of dimension $m$ to dimension $n$. This means the transformation collapses or creates dimensions. Such a transformation is not uniquely reversible.

For a matrix to be invertible it must represent a **bijection** (a function that maps every member of a set onto exactly one member of another set).

### Singular and non-singular matrices

A matrix with det\(A\)=0 is called **singular** and has no inverse.

A matrix which is invertible is called **non\-singular**. 

The geometric intuition for this is simple. Going back to the paralleogram model, a matrix with zero determinant has at least one zero eigenvalue. This means that at least one of the dimensions of the parallelepiped has been squashed to nothing at all. Therefore it is impossible to reverse the transformation, because information was lost in the forward transform. 

All of the original dimensions must be preserved in a linear map for inversion to be meaningful; this is the same as saying det\(A\) ≠ 0.


#### Special cases

Just as for multiplication, there are many special kinds of matrices for which much faster inversion is possible. These include, among many others:

* orthogonal matrix \(rows and columns are all orthogonal unit vectors\): O\(1\), A<sup>-1</sup>= A<sup>T</sup>
* diagonal matrix \(all non\-diagonal elements are zero\): O\(n\), A<sup>-1</sup> = 1/A \(i.e. the reciprocal of the diagonal elements of A\).
* positive\-definite matrix: O\(n<sup>2</sup>\)$ via the *Cholesky decomposition*. We won't discuss this further.
* triangular matrix \(all elements either above or below the main diagonal are zero\): O\(n<sup>2</sup>\), trivially invertible by **elimination algorithms**.

#### Special cases: orthogonal matrices

An **orthogonal matrix** is a special matrix form that has A<sup>T</sup>=AA<sup>-1</sup> that is the transpose and the inverse are equivalent. All of its component eigenvectors are **orthogonal** to each other \(at 90 degrees; have an inner product of 0\), and all of its eigenvalues are 1 or \-1. An orthogonal matrix transforms a cube to a cube. It has a determinant of 1 or \-1. Any purely rotational matrix is an orthogonal matrix. 

Orthogonal matrices can be inverted trivially, since transposition is essentially free in computational terms.

#### Special cases: Diagonal matrices

The inverse of a diagonal matrix is another diagonal matrix whose diagonal elements are the reciprocal of each of the diagonal elements of the original matrix:  
\(A<sub>ii</sub>\)<sup>-1</sup> = 1 / A<sub>ii</sub>

### Issue with inversion of sparse matrices

The inverse of a **sparse matrix** is in general **not sparse**; it will \(most likely\) be dense. This means that sparse matrix algorithms virtually never involve a direct inverse, as a sparse matrix could easily be 1,000,000 x 1,000,000, but with maybe only a few million non\-zero entries, and might be stored in a few dozen megabytes. The inverse form would have 1,000,000,000,000 entries and require a terabyte or more to store!

### Linear systems

One way of looking at matrices is as a collection of weights of components of a vector. 

The coefficients of the matrix represent the weighting to be applied.

### Solving linear systems

The solution of linear systems is apparently simple for cases where A is square. If Ax=y, then left\-multiplying both sides by A<sup>-1</sup> we get 

x = A<sup>-1</sup>y

This only works for square matrices, as A<sup>-1</sup> is not defined for non\-square matrices. This means that x and y must have the same number of dimensions.

### Approximate solutions for linear systems

In practice, linear systems are almost never solved with a direct inversion. The numerical problems in inverting high dimensional matrices will make the result highly unstable, and tiny variations in y might lead to wildly different solutions for x.

Instead, linear systems are typically solved iteratively, either using specialised algorithms based on knowledge of the structure of the system, or using **optimisation**, which will be the topic of the next Unit.

These algorithms search the possible space of x to find solutions that minimise
\|Ax\-y\|<sub>2</sub><sup>2</sup> by adjusting the value of x repeatedly.

The reason these iterative approximation algorithms can work when inversion is numerically impossible is that they only have to solve for one *specific* pair of vectors x, y. They do not have to create an inversion A<sup>-1</sup> that inverts the problem for all possible values of y, just the specific y seen in the problem. This problem is much more constrained and therefore much more stable.


### Singular value decomposition

Eigendecompositions only apply to diagonalizable matrices; which are a subset of square matrices. But the ability to "factorise" matrices in the way the eigendecomposition does is enormously powerful, and there are many problems which have non-square matrices which we would like to be able to decompose.

The **singular value decomposition** \(SVD\) is a general approach to decomposing any matrix A. It is the powerhouse of computational linear algebra.

The SVD produces a decomposition which splits ***ANY*** matrix up into three matrices:

A = U Σ V<sup>T</sup>

where 
* A is any m x n matrix, 
* U is a **square unitary** m x m matrix, whose columns contain the **left singular vectors**,
* V is an **square unitary** n x n matrix, whose columns contain the **right singular vectors**,
* Σ is a diagonal m x n matrix, whose diagonal contains the **singular values**.

A **unitary** matrix is one whose conjugate transpose is equal to its inverse. If A is real, then U and V will be **orthogonal** matrices \(U<sup>T</sup> = U<sup>-1</sup>\), whose rows all have unit norm and whose columns also all have unit norm.

The diagonal of the matrix Σ is the set of **singular values**, which are closely related to the eigenvalues, but are *not* quite the same thing \(except for special cases like when A is a positive semi\-definite symmetric matrix\)! The **singular values** are always positive real numbers.

### Relation to eigendecomposition

The SVD is the same as:   
* taking the eigenvectors of A<sup>T</sup>A to get U
* taking the square root of the *absolute* value of the eigenvalues λ<sub>i</sub> of A<sup>T</sup>A to get Σ<sub>i</sub> = sqrt\(\|λ<sub>i</sub>\|\)
* taking the eigenvectors of AA<sup>T</sup> to get V<sup>T</sup>

### Special case: symmetric, positive semi-definite matrix

For a symmetric, positive semi\-definite matrix A, the eigenvectors are the columns of U or the columns of V. The eigenvalues are in Σ.

### SVD decomposes any matrix into three matrices with special forms

Special forms of matrices, like orthogonal matrices and diagonal matrices, are much easier to work with than general matrices. This is the power of the SVD.

* U is orthogonal, so is a pure rotation matrix,
* Σ is diagonal, so is a pure scaling matrix, 
* V is orthogonal, so is a pure rotation matrix.

#### Rotate, scale, rotate

*The SVD splits any matrix transformation into a rotate\-scale\-rotate operation.*

### Using the SVD

Many matrix operations are trivial once we have the factorisation of the matrix into the three components. But some can only be performed on *on certain types of matrices*.

###  Fractional powers

We can use the SVD to compute interesting matrix functions like the square root of a matrix A<sup>1/2</sup>. In fact, we can use the SVD to raise a matrix to any power, *in a single operation*, provided it is a **square symmetric matrix**. 

We can use the SVD to:

* raise a matrix to a fractional power, e.g. A<sup>1/2</sup>, which will "part do" an operation,
* invert a matrix: A<sup>-1</sup>, which will "undo" the operation.

The rule is simple: to do any of these operations, ignore U and V \(which are just rotations\), and apply the function to the singular values elementwise:

A<sup>n</sup> = V Σ<sup>n</sup> U<sup>T</sup> 

For a symmetric matrix, this is the same as:

A<sup>n</sup> = U Σ<sup>n</sup> V<sup>T</sup> 

**Note: A<sup>1/2</sup> is not the elementwise square root of each element of A!** 

Rather, we must comput the elementwise square root of Σ, then compute A<sup>1/2</sup> = U Σ<sup>1/2</sup> V<sup>T</sup>.

### Inversion

We can efficiently invert a matrix once it is in SVD form. For a non\-symmetric matrix, we use:

A<sup>-1</sup> = V Σ<sup>-1</sup> U<sup>T</sup>

This can be computed in O\(n\) time because Σ<sup>-1</sup> can be computed simply by taking the reciprocal of each of the diagonal elements of Σ. 

*N.B. As a consequence, we now know that computing the SVD must take O\(n<sup>3</sup>\) time for square matrices, since inversion cannot be achieved faster than O\(n<sup>3</sup>\).*

### Pseudo\-inverse

We can also pseudo-invert a matrix: A<sup>\+</sup>, which will approximately "undo" the operation, even when A isn't square.

This means we can solve \(approximately\) systems of equations where the number of input variables is different to the number of output variables. The **pseudo\-inverse** or **Moore\-Penrose pseudo\-inverse** generalises inversion to \(some\) non\-square matrices. 

We can find approximate solutions for x in the equation:

Ax = y 

or in fact simultaneous equations of the type

AX = Y

The pseudo\-inverse of A is just 

A<sup>\+</sup> = V Σ<sup>-1</sup> U<sup>T</sup>

which is the same as the standard inverse computed via SVD, but taking care that Σ is the right shape - appropriate zero padding is required! Fortunately, this is taken care of by the Numpy method `np.linalg.pinv`.

### Fitting lines/planes using the pseudo-inverse

Suppose we have a data set which consists of multiple examples of x and y, stored in data matrices X and Y, where each example forms a row. We'd like to fit a line/plane to this data such that we can predict an individual y\-value using the formula y = Ax, or lots of y\-values using the formula Y = AX. The equation for the line/plane is represented by the matrix A, and this is what we want to find.

We can find an approximate solution for $A$ by using the pseudo-inverse on our existing data set:

A = X<sup>\+</sup>Y

This allows us to fit a line/plane to a collection of data points, even when we have many more points than the number of dimensions required to specify the line/plane \(standard inversion would require us to have exactly the same number of data points as dimensions, in order to obtain an exact solution\). A system of equations where there are more inputs than outputs is called "overdetermined". The pseuo\-inverse allows us to solve overdetermined problems like this one.

### Rank of a matrix

The **rank** of a matrix is equal to the number of non\-zero singular values. 

* If the number of non-zero singular values is equal to the size of the matrix, then the matrix is **full rank**. 
* A full rank matrix has a non\-zero determinant and will be invertible. 
* The rank tells us how many dimensions the parallelotope that the transform represents will have. 
* If a matrix does not have full rank, it is **singular** \(non\-invertible\) and has **deficient rank**.
* If the number of non-zero singular values is much less than the size of the matrix, the matrix is **low rank**.

For example, a 4 x 4 matrix with rank 2 will take vectors in 4dim and output vectors in a 2dim subspace \(a plane\) of 4dim. The orientation of that plane will be given by the first two eigenvectors of the matrix.

### Condition number of a matrix

The **condition number** number of a matrix is the ratio of the largest singular value to the smallest. 

* This is only defined for full rank matrices. 
* The condition number measures how sensitive inversion of the matrix is to small changes.
* A matrix with a small condition number is called **well-conditioned** and is unlikely to cause numerical issues. 
* A matrix with a large condition number is **ill-conditioned**, and numerical issues are likely to be significant. 

An ill-conditioned matrix is almost singular, so inverting it will lead to invalid results due to floating point roundoff errors.

### Relation to singularity

A **singular** matrix A is un\-invertible and has det\(A\)=0. Singularity is a binary property, and is either true or false. 

**Rank** and **condition numbers** extend this concept.

* We can think of **rank** as measuring "how singular" the matrix is, i.e. how many dimensions are lost in the transform.
* We can think of the **condition number** as measuring how close a non-singular matrix is to being singular. A matrix which is nearly singular may become effectively singular due to floating point roundoff errors.

### Applying decompositions

#### Whitening a data set

**Whitening** removes all linear correlations within a dataset. It is a *normalizing* step used to standardise data before analysis. This will be covered in the lab.

Given a data set stored in matrix X, we can compute:

X<sup>w</sub> = X - μ Σ<sup>-1/2</sup> 

where μ is the **mean vector**, i.e. a row vector containing the mean of each column in X, and Σ is the **covariance matrix** \(not the matrix of singular values\).

This equation takes each column of X and subtracts the mean of that column from every element in the column, so that each column is centred on zero. Then it multiplies by the inverse square root of the covariance matrix, which is a bit like dividing each column of X by its standard deviation to normalise the spread of values in each column. However, it is more rigorous than that, because it also removes any correlations between the columns.

To summarise, whitening does the following:

* centers the data around its mean, so it has **zero mean**.
* "squashes" the data so that its distribution is spherical and has **unit covariance**.

Removing the mean is easy. The difficult bit is computing the inverse square root of the covariance matrix. N.B. this is *definitely not* the inverse square root of the elements of the covariance matrix!

Fortunately, we can compute it easily by taking the SVD of the covariance matrix. We compute the inverse square root in one step, by taking the reciprocal of the square root of each of the singular values and then reconstructing. 

The whitened version of the data set will have all linear correlations removed. This is an important preprocessing step when applying machine learning and statistical analysis algorithms. 

## Week 6

### What is optimisation?

**Optimisation** is the process of adjusting things to make them better. In computer science, we want to do this *automatically* by a algorithm. An enormous number of problems can be framed as optimisation, and there are a plethora of algorithms which can then do the automatic adjustment *efficiently*, in that they find the best adjustments in few steps. In this sense, *optimisation is search*, and optimisation algorithms search efficiently using mathematical structure of the problem space.

Optimisation is at the heart of machine learning; it is a critical part of all kinds of manufacturing and industrial processes, from shipbuilding to circuit design; it can even be used to automatically make scatterplot graphs easier to read.

### One algorithm to rule them all: no special cases

Optimisation algorithms allow us to apply *standard algorithms* to an enormous number of problems. We don't have special cases for every specific problem; instead we formulate the problems so that generic algorithms can solve them. As a consequence, to apply optimisation algorithms, the problems must be specified formally. There is a real art in specifying problems so that optimisation can tackle them.

### Parameters and objective function


There are two parts to an optimisation problem:

* **parameters**: the things we can adjust, which might be a scalar or vector or other array of values, denoted θ. The parameters exist in a **parameter space** -- the set of all possible configurations of parameters denoted Θ. This space is often a **vector space** like n, but doesn't need to be. For example, the set of all knob/slider positions on the synthesizer panel above could be considered points in a subset of a vector space. If the parameters do lie in a vector space, we talk about the **parameter vector** θ.


* **the objective function**: a function that maps the parameters onto a *single numerical measure* of how good the configuration is. L\(θ\).  The output of the objective function is a single scalar. The objective function is sometimes called the *loss function*, the *cost function*, *fitness function*, *utility function*, *energy surface*, all of which refer to \(roughly\) the same concept. It is a quantitative \("objective"\) measure of "goodness".

*The desired output of the optimisation algorithm is the parameter configuration that minimises the objective function.*

Writing this mathematically, this is the min\(\) (the argument that produces the minimum value) of the objective function:


θ<sup>\*</sup> = min\(θ in Θ\) L\(θ\) 


* θ<sup>\*</sup> is the configuration that we want to find; the one for which the objective function is lowest. 
* Θ is the set of all possible configurations that θ could take on, e.g. N. 

Most optimisation problems have one more component:

* **constraints**: the limitations on the parameters. This defines a region of the parameter space that is feasible, the **feasible set** or **feasible region**. For example, the synthesizer above has knobs with a fixed physical range, say 0-10; it isn't possible to turn them up to 11. Most optimisation problems have constraints of some kind; 

> "design a plane *\(adjust parameters\)* that flies as fast as possible *\(objective function\)*, and costs less than `$180M` *\(constraints\)*.

We usually think of the objective function as a **cost** which is *minimised*. Any maximisation problem can be reframed as a minimisation problem by a simple switch of sign, so this does not lose generality. If if we wanted to optimise the knob settings on our synthesizer to make a really good piano sound \("maximise goodness"\), we could instead frame this as a problem of minimising the difference between the sound produced and the sound of a piano. We would, of course, need to have a **precise** way of measuring this difference; one that results in a single real number measure of cost.

### Minimising differences

As in this example, it is common to have express problems in a form where the objective function is a  **distance between an output and a reference is measured**. Not every objective function has this form, but many do.

That is, we have some function y<sup>'</sup> = f\(x;θ\) that produces an output from an input x governed by a set of parameters θ, and we measure the difference between the output and some reference y \(e.g. using a vector norm\):


L\(θ\) = \|y<sup>'</sup> \- y\| = \|f(x};θ\) - y\| 

This is very common in **approximation problems**, where we want to find a function that approximates a set of measured observations. This is the core problem of machine learning.

Note that the notation f\(x;θ\) just means that the output of f depends both on some \(vector\) input x and on a parameter vector θ. Optimisation only ever adjusts θ, and the vector x is considered fixed during optimisation \(it might, for example, represent a collection of real\-world measurements\). 

### Evaluating the objective function

It may be **expensive** to evaluate the objective function. For example:
* the computation might take a long time \(invert a 10000x10000 matrix\);
* or it might require a real\-world experiment to be performed \(do the users like the new app layout?\);
* or it might be dangerous \(which wire on the bomb should I cut next?\);
* or it might require data that must be bought and paid for \(literally expensive\). 

In all cases, it will take some computational power to evaluate the objective function, and therefore will have a time cost.

This means that a *good* optimisation algorithm will find the optimal configuration of parameters with few queries \(evaluations of the objective function\). To do this, there must be mathematical **structure** which can help guide the search. Without any structure at all, the best that could be done would be to randomly guess parameter configurations and choose the lowest cost configuration after some number of iterations. This isn't typically a feasible approach.

### Discrete vs. continuous

If the parameters are in a continuous space \(typically R<sup>n</sup>\), the problem is one of **continuous optimization**; if the parameters are discrete, the problem is **discrete optimization**. Continuous optimisation is usually easier because we can exploit the concept of **smoothness** and **continuity**.

#### Properties of optimisation

Every optimisation problem has two parts:
* **Parameters**, the things that can be adjusted.
* **Objective function**, which measures how good a particular set of parameters are.

An optimisation problem usually also has:
* **Constraints**, that define the feasible set of parameters.

The **objective function** is a function *of the parameters* which returns a *single scalar value*, representing how good that parameter set is. 

### Throwing a stone

For example, if I wanted to optimise how far I could throw a stone, I might be able to adjust the throwing angle. This is the *parameter* I could tweak \(just one parameter θ = \[α\], in this case\). 

The objective function must be a function which depends on this parameter. I would have to *simulate* throwing the ball to work out how far it went and try and make it go further and further.

### Focus: continuous optimisation in real vector spaces

This course will focus on optimisation of continuous problems in n. That is θ in n = \[θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>\] and the optimisation problem is one of:

θ<sup>\*</sup> = min\(θ in \n\) L\(θ\), \(subject to constraints\)

This it the problem of searching a continuous vector space to find the point where L\(θ\) is smallest. We will typically encounter problems where the objective function is *smooth* and *continuous* in this vector space; note that the parameters being elements of a continuous space does not necessarily imply that the objective function is continuous in that space.

Some optimisation algorithms are **iterative**, in that they generate successively better approximations to a solution. Other methods are **direct**, like linear least squares \(which we'll briefly discuss\), and involving finding a minimum exactly in one step. We will focus primarily on **iterative, approximate** optimisation in this course.

#### A function of space

The objective function maps points in space to values; i.e. it defines a curve/surface/density/etc. which varies across space. We want to find, as quickly as possible, a point in space where this is as small as possible, without going through any "walls" we have defined via constraints.

### Geometric median: optimisation in n2

* **Problem** Find the median of a `>1D` dataset. The standard median is computed by sorting and then selecting the middle element \(with various rules for even sized datasets\). This doesn't work for higher dimensions, and there is no straightforward direct algorithm. But there is an easy definition of the median: it is the vector that minimises the sum of distances to all vectors in the dataset.

A very simple optimisation example is to find a point that minimises the distance to a collection of other points \(with respect to some norm\). We can define:

* **parameters** θ=\[x, y, ...\], a position  in 2D. 
* **objective function** the sum of distances between a point and a collection of target points x<sub>i</sub>:

L\(θ\) \= Σ \|\|θ \- x<sub>i</sub>\|\|<sup>2</sup>

This will try and find a point in space \(represented as θ\) which minimises the distances to the target points. We can solve this, starting from some random initial condition \(guess for θ\):

### An example of optimisation in N

We can work in higher dimensions just as easily. A slightly different problem is to try and find a layout of points in such that the points are **evenly spaced** \(with respect to some norm\). In this case we have to optimise a whole collection of points, which we can do by rolling them all up into a single parameter vector.

We can define:

* **parameters** θ=\[x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>, y<sub>2</sub>, ...\], an array of positions of points in 2D. Note: we have "unpacked" a sequence of 2D points into a higher dimensional vector, so that a *whole configuration* of points is a single point in a vector space.
* **loss function** the sum of squares of differences between the Euclidean pairwise distances between points and some target distance:

\Σ for i Σ for j \(α \- \|\|x<sup>i</sup> \- x<sup>j</sup>\|\|<sup>2</sup>\)<sup>2</sup>

This will try and find a configuration of points that are all α units apart.

### Constrained optimisation

If a problem has constraints on the parameters beyond purely minimising the objective function then the problem is **constrained optimisation**.

A constrained optimisation might be written in terms of an equality constraint:    
θ<sup>\*</sup> = min\(θ in Θ\) L\(θ\), subject to c\(θ\) = 0  
or an inequality:  
θ<sup>\*</sup> = min\(θ in Θ\) L\(θ\) ,subject to c\(θ\) ≤ 0   
where c\(θ\) is a function that represents the constraints.

* An **equality** constraint can be thought of as constraining the parameters to a *surface*, to represent a tradeoff. For example, c\(θ\) =\|θ\|<sup>2</sup>\-1 forces the parameters to lie on the surface of a unit sphere. An equality constraint might be used when trading off items where the total value must remain unchanged \(e.g. the payload weight in a satellite might be fixed in advance\).

* An **inequality** constraint can be thought of as constraining the parameters to a *volume*, to represent bounds on the values. For example, c\(θ\) =\|θ\|<sub>∞</sub>\-10 forces the parameters to lie within a box extending \(\-10, 10\) around the origin -- perhaps the range of the knobs on the synthesizer.

#### Common constraint types

A **box constraint** is a simple kind of constraint, and is just a requirement that θ lie within a box inside $R^n$; for example, that every element 0<θ<sub>i</sub><1 \(all parameters in the positive unit cube\) or θ<sub>i</sub>\>0 \(all parameters in the positive **orthant**\). This is an inequality constraint with a simple form of c\(θ\). Many optimisation algorithms support box constraints.

A **convex constraint** is another simple kind of constraint, where the constraint is a collection of inequalities on a convex sum of the parameters θ. Box constraints are a specific subclass of convex constraints. This is equivalent to the feasible set being limited by the intersection of many of **planes/hyperplanes** \(possibly an infinite number in the case of curved convex constraints\).

**Unconstrained optimization** does not apply any constraints to the parameters, and any parameter configuration in the search space is possible. In many problems, pure unconstrained optimisation will lead to unhelpful results \("the airplane will get the best lift if the wing is two hundred miles long" -- which might be true but impossible to construct\).


### Constraints and penalties

Unconstrained optimisation rarely gives useful answers on its own. Consider the example of the airfoil. Increasing lift might be achieved by making the airfoil length longer and longer. At some point, this might become physically impossible to build.

Although we often represent $θ$ as being in $\real^N$, the feasible set is typically not the entire vector space. There are two approaches to deal with this:

#### Constrained optimisation

* Use an optimisation algorithm that supports hard constraints inherently. This is straightforward for certain kinds of optimisation, but trickier for general optimisation.Typically constraints will be specified as either a **convex region** or a simple \(hyper\)rectangular region of the space \(a **box constraint**\).
* **Pros**: 
    * Guarantees that solution will satisfy constraints.
    * May be able to use constraints to speed up optimisation.
* **Cons**: 
    * may be less efficient than unconstrained optimization. 
    * Fewer algorithms available for optimisation.
    * may be hard to specify feasible region with the parameters available in the optimiser.

#### Soft constraints

* Apply penalties to the objective function to "discourage" solutions that violate the constraints. This is particularly appropriate if the constraints really are soft (it doesn't perhaps matter if the maximum airfoil length is 1.5m or 1.6m, but it can't be 10m). In this case, the penalties are just terms added to the objective function. The optimiser stays the same, but the objective function is modified.

L(θ<sup>'</sup>') = L\(θ\) + λ\(θ\), where λ\(θ\) is a **penalty function** with an increasing value as the constraints are more egregiously violated.

* **Pros**
    * any optimiser can be used
    * can deal with *soft* constraints sensibly
* **Cons:** 
    * may not respect important constraints, particularly if they are very sharp
    * can be hard to formulate constraints as penalties
    * cannot take advantage of efficient search in constrained regions of space

### Relaxation of objective functions

It can be much harder to solve discrete optimization and constrained optimization problems efficiently; some algorithms try and find similar continuous or unconstrained optimization problems to solve instead. This is called **relaxation**; a **relaxed** version of the problem is solved instead of the original hard optimization problem. For example, sometimes the constraints in a problem can be absorbed into the objective function, to convert a constrained problem to an unconstrained problem. 

#### Penalisation

**Penalisation** refers to terms which augment an objective function to minimise some other property of the solution, typically to approximate constrained optimisation. This is widely used in approximation problems to find solutions that **generalise well**; that is which are tuned to approximate some data, but not *too* closely.

This is a relaxation of a problem with hard constraints \(which needs specialised algorithms\) to a problem with a simple objective function which works with any objective function. If you have encountered **Lagrange multipliers** before, these are an example of a relaxation of hard constraints to penalty terms.

### Penalty functions

A **penalty function** is just a term added to an objective function which will disfavour "bad solutions". 

We can return to the stone throwing example, and extend our model. Say
I can control the angle of a stone throw; perhaps I can also control how hard I throw it. But there is a maximum limit to my strength. This is a constraint \(an inequality constraint, which limits the maximum value of the strength parameter\).

* Objective function: how far away does the stone land? L\(θ\) = throw_distance\(θ\)
* Parameters: angle of the throw $\alpha$ and strength of the throw v \(exit velocity\), θ=\[α, v\]$
* Constraint: strength of throw 0 ≤ v ≤ v<sub>k</sub>, more than zero and less than some maximum strength.

There are two options:
* Use a constrained optimisation algorithm, which will not even search solutions which exceed the maximum strength.
* Change the objective function to make over\-strenuous throwing unacceptable.

#### Option 1: constrained optimisation

Use a \(pre\-existing\) algorithm which already supports constraints directly. Guarantees solutions will lie inside bounds.

#### Option 2: add a penalty term

L'\(θ\) = L\(θ\) \+ λ\(theta\)


### Properties of the objective function

#### Convexity, global and local minima

An objective function may have **local minima**. A **local minimum** is any point where the objective functions increases in every direction around that point \(that parameter setting\). Any change in the parameters at that point increases the objective function.

An objective function is **convex** if it has a *single, global minimum*. For example, every quadratic function is a parabola \(in any number of dimensions\), and thus has exactly one minimum. Other functions might have regions that have local minimums but which **aren't** the smallest possible value the function could take on.

**Convexity** implies that finding any minimum is equivalent to finding the global minimum -- the guaranteed best possible solution.  This minimum is the global minimum. In a convex problem, if we find a minimum, we can stop searching. If we can show there *is no minimum*, we can also stop searching.

#### Convex optimisation

If the objective function is **convex** *and* any constraints form convex portions of the search space, then the problem is **convex optimisation**. There are very efficient methods for solving convex optimisation problems, even with tens of thousands of variables. These include: 

* the constraints and objective function are linear \(**linear programming**\)
* quadratic objective function and linear constraints \(**quadratic programming**\)
* or a some specialised cases \(**semi\-quadratic programming**, **quadratically constrained quadratic program**\). 

These are incredible powerful algorithms for solving these specific classes of optimisation problems. 

Nonconvex problems require the use of **iterative** methods \(although ways of *approximating* nonconvex problems with convex problems do exist\).

#### Continuity

An objective function is **continuous** if for some very small adjustment to $θ$ there is an *arbitrarily* small change in L\(θ\). This means that there will never be "nasty surprises" if we move slowly enough through the space of θ; no sudden jumps in value.

If a function is discontinuous, local search methods are not guaranteed to converge to a solution. Optimisation for discontinuous objective functions is typically much harder than for continuous functions. This is because there could be arbitrary changes in the objective function for any adjustment to the parameters.

Continuity is what can make continuous optimisation easier than discrete optimisation. As we will see next week, being continuous and **differentiable** makes continuous optimisation even more powerful.

### Algorithms
#### Direct convex optimisation: least squares

Sometimes we have an optimisation problem which we can specify such that the solution can be computed in one step. An example is **linear least squares**, which solves objective functions of the form:
    
min<sup>x</sup>\(L\(x\)\) = \|Ax\-y\|<sub>2</sub><sup>2</sup>, 

that is, it finds x that is closest to the solution Ax=y in the sense of minimising the squared L2 norm. The squaring of the norm just makes the algebra easier to derive. 

This equation is **convex** -- it is a quadratic function and even in multiple dimensions it must have a single, global minimum, which can be found directly. The reason we **know** it is convex is that it has no terms with powers greater than 2 \(no x<sup>3</sup> etc.\) and so is quadratic. Quadratic functions only ever have zero or one minimum.

### Line fitting
We will examine this process for the simplest possible **linear regression** example: finding gradient $m$ and offset $c$ for the line equation y=mx\+c such that the squared distance to a set of observed \(x,y\) data points is minimised. This is a search over the θ=\[m,c\] space; these are the parameters. The objective function is L\(θ\) = ∑ \(y \- mx<sub>i</sub>\+c)<sup>2</sup>, for some known data points \[x<sub>0</sub>, y<sub>0</sub>\], \[x<sub>1</sub>, y<sub>1</sub>\], etc.

We can solve this directly using the **pseudo-inverse** via the SVD. This is a problem that can be solved directly in one step.


### Iterative optimisation

**Iterative optimisation** involves making a series of steps in parameter space. There is a **current parameter vector** \(or collection of them\) which is adjusted at each iteration, hopefully decreasing the objective function, until optimisation terminates after some **termination criteria** have been met.

Iterative optimisation algorithm:

1. choose a starting point x<sub>0</sub>
1. while objective function changing
    1. adjust parameters
    1. evaluate objective function
    1. if better solution found than any so far, record  it
1. return best parameter set found


### Regular search: grid search

**Grid search**, is a straightforward but inefficient optimisation algorithm for multidimensional problems. The parameter space is simply sampled by equally dividing the feasible set in each dimension, usually with a fixed number of divisions per dimension.

The objective function is evaluated at each $θ$ on this grid, and the lowest loss θ found so far is tracked. This is simple, and can work for 1D optimisation problems. It is sometimes used to optimise *hyperparameters* of machine learning problems where the objective function may be complex but finding the absolute minimum isn't essential.


### Revenge of the curse of dimensionality

> Why bother optimising? Why not just search every possible parameter configuration? 

Even in relatively small parameter spaces, and where the objective function is known to be smooth this doesn't scale well.  Simply divide up each dimension into a number of points \(maybe 8\), and then try every combination on the grid of points that this forms, choosing the smallest result.


While this is fine in 1D \(just check 8 points\) and 2D \(just check 64 points\), it breaks down completely if you have a 100 dimensional parameter space. This would need evaluations of the objective function! The synthesizer above has around 100 dimensions, as an example.

Even just 3 points in each dimension is totally unreasonable.
        
### Density of grid search

If the objective function is not very smooth, then a much denser grid would be required to catch any minima.

Real optimisation problems might have hundreds, thousands or even billions of parameters \(in big machine learning problems\). Grid search and similar schemes are *exponential* in the number of dimensions of the parameter space.

#### Pros

* Works for any continuous parameter space.
* Requires no knowledge of the objective function.
* Trivial to implement.

#### Cons

* **Incredibly** inefficient
* Must specify search space bounds in advance.
* Highly biased to finding things near the "early corners" of the space.
* Depends heavily on number of divisions chosen.
* Hard to tune so that minima are not missed entirely.


### Hyperparameters

Grid seach depends on the **range** searched and the spacing of the **divisions** of the grid. Most optimisation algorithms have similar properties that can be tweaked.

These properties, which affect the way in which the optimiser finds a solution, are called **hyperparameters**. They are not parameters of the objective function, but they do affect the results obtained. 

A perfect optimiser would have no hyperparameters -- a solution should not depend on how it was found. But in practice, all useful optimisers have some number of hyperparameters which will affect their performance. Fewer hyperparameters is usually better, as it is less cumbersome to tune the optimiser to work.

### Simple stochastic: random search

The simplest such algorithm, which makes *no* assumptions other than we can draw random samples from the parameter space, is **random search**.  

The process is simple:  
* Guess a random parameter θ
* Check the objective function L\(θ\)
* If L\(θ\)<L\(θ<sup>\*</sup>) \(the previous best parameter θ<sub>\*</sub>\), set θ<sub>\*</sub>=θ

There are many possibilities for a termination condition, such as stopping after a certain number of iterations after the last change in the best loss. The simple code below uses a simple fixed iteration count and therefore makes no guarantee that it finds a good solution at all.

### Pros
* Random search cannot get trapped in local minima, because it uses no local structure to guide the search. 
* Requires no knowledge of the structure of the objective function \- not even a topology.
* Very simple to implement.
* Better than grid search, almost always.

### Cons
* *Extremely inefficient* and is usually only appropriate if there is no other mathematical structure to exploit.
* Must be possible to randomly sample from the parameter space \(usually not a problem, though\).
* Results do not necessarily get better over time. Best result might be found in the first step or a million steps later. There is no way to predict how the optimisation will proceed.

### bogosort

The \(joke\) sorting algorithm **bogosort** uses random search to sort sequences. The algorithm is simple:

* randomise the order of the sequence
* check if it is sorted
    * if it is, stop; otherwise, repeat

This is amazingly inefficient, taking O\(n!\) time to find a solution, which is even worse than exponential time. In this application, the parameter space \(all possible orderings\) is so huge that random search is truly hopeless. It is particularly poor because of the binary nature of the loss function -- either it is perfect, or it is disregarded, so we will never even get approximately correct results.

However, it is a correct implementation of a sorting algorithm.

### Metaheuristics

There are a number of standard **meta-heuristics** than can be used to improve random search.

These are:  
* **Locality** which takes advantage of the fact the objective function is likely to have similar values for similar parameter configurations. This assumes *continuity* of the objective function.
* **Temperature** which can change the rate of movement in the parameter space over the course of an optimisation. This assumes the existence of local optima.
* **Population** which can track multiple simultaneous parameter configurations and select/mix among them.
* **Memory** which can record good or bad steps in the past and avoid/revisit them.

#### Locality

**Local search** refers to the class of algorithms that make *incremental* changes to a solution. These can be much more efficient than random search or grid search when there is some continuity to the objective function. However, they are subject to becoming trapped in **local minima**, and not reaching the **global minimum**. Since they are usually exclusively used for nonconvex problems, this can be a problem.

This implies that the output of the optimisation depends on the **initial conditions**. The result might find one local minimum starting from one location, and a different local minimum from another starting parameter set.

Local search can be thought of forming **trajectory** \(a path\) through the parameter space, which should hopefully move from high loss towards lower loss.

#### Hill climbing: local search

**Hill climbing** is a modification of random search which assumes some topology of the parameter space, so that there is a meaningful concept of a **neighbourhood** of a parameter vector; that we can make incremental changes to it. Hill climbing is a form of **local search**, and instead of drawing samples randomly from the parameter space, randomly samples configurations *near* the current best parameter vector. It makes incremental adjustments, keeping transitions to neighbouring states only if they improve the loss.

**Simple hill climbing** adjusts just one of the parameter vector elements at a time, examining each "direction" in turn, and taking a step if it improves things. **Stochastic hill climbing** makes a random adjustment to the parameter vector, then either accepts or rejects the step depending on whether the result is an improvement.

The name *hill climbing* comes from the fact that the algorithm randomly wanders around, only ever taking uphill (or downhill, for minimisation) steps. Because hill climbing is a **local search** algorithm, it is vulnerable to getting stuck in local minima. Basic hill climbing has no defence against minima and will easily get trapped in poor solutions if they exist. Simple hill climbing can also get stuck behind **ridges** and all forms of hill climbing struggle with **plateaus** where the loss function changes slowly. 

##### Pros
* Not much more complicated than random search
* Can be *much* faster than random search

##### Cons
* Hard to choose how much of an adjustment to make
* Can get stuck in minima
* Struggles with objective function regions that are relatively flat
* Requires that the objective function be \(approximately\) continuous

#### Simulated annealing

**Simulated annealing** extends hill-climbing with the ability to sometimes randomly go uphill, instead of always going downhill. It uses a **temperature schedule** that allows more uphill steps at the start of the optimisation and fewer ones later in the process. This is used to overcome ridges and avoid getting stuck in local minima.

The idea is that allowing random "bad jumps" early in a process can help find a better overall configuration.

##### Accepting you have to go uphill sometimes

Simulated annealing uses the idea of **acceptance probability**. Instead of just accepting any random change that decreases the loss, we randomly accept some proportion of jumps that might temporarily increase the loss, and slowly decrease the proportion of these over time.

Given the current loss l = L\(θ\) and a proposed new loss l<sup>'</sup> = L\(θ\+Δθ\), where Δθ represents a random perturbation of θ, we can define a probability P\(l, l<sup>'</sup>, T\(i\)\) which is the probability of jumping from θ to Δθ at iteration i.

A common rule is:
* P\(l, l<sup>'</sup>, T\(i\)\)=1 if l<sup>'</sup> < l, i.e. always go downhill.
* P\(l,l<sup>'</sup>,T\(i\)) = e<sup>{\-(l\-l<sup>'</sup>)</sup>T\(i\) i.e. take uphill jumps if the relative decrease is small.

T\(i\) is typically an exponentially decaying function of the iteration number, so that large jumps are accepted at the start, even if they go *way* uphill, but there is a decreasing tendency to make uphill jumps as time goes on.

For example, T\(i\) = e<sup>\-i / r</sup>, where i is the iteration number, r is the cooling rate and T is the temperature.

#### Pros

* Much less sensitive to getting trapped in minima than hill climbing
* Easy to implement
* Empirically very effective
* Fairly effective even in mixed continuous/discrete settings.

#### Cons

* Depends on good choice for the temperature schedule and neighbourhood function, which are extra free parameters to worry about.
* No guarantees of convergence
* Slow if the uphill steps are not actually needed

### More complicated example: finding evenly spaced points

This isn't very impressive for the line fitting, which is a very simple convex function; there are no local minima to get trapped in. We can look at the problem of finding a collection of points that are evenly spaced. This is non\-convex \(and has an infinite number of equal minima\), and much harder to solve than fitting a line to some points.

This is a task particularly suited to simulated annealing\-style approaches.

## Population

Another nature\-inspired variant of random search is to use a **population** of multiple competing potential solutions, and to apply some analogue of **evolution** to solving optimisation. This involves some of:

* **mutation** \(introducing random variation\)
* **natural selection** \(solution selection\)
* **breeding** \(interchange between solutions\)

This class of algorithms are often called **genetic algorithms** for obvious reasons. All genetic algorithms maintain some population of potential solutions \(a set of vectors θ<sub>1</sub>, θ<sub>2</sub>, θ<sub>3</sub>, ...\), and some rule which is used to preserve some members of the population and cull others. The parameter set is referred to as the **genotype** of a solution.

Simple population approaches simply use small random perturbations and a simple selection rule like "keep the top 25% of solutions, ordered by loss". Each iteration will perturb the solutions slightly by random mutation, cull the weakest solutions, then copy the remaining "fittest" solutions a number of times to produce the offspring for the next step. The population size is held constant from iteration to iteration. This is just random local search with population. The idea is that this can explore a larger area of the space than simple local search and maintain multiple possible hypotheses about what might be good during that time.

### Genetic algorithms: population search
#### Pros

* Easy to understand and applicable to many problems.
* Requires only weak knowledge of the objective function
* Can be applied to problems with both discrete and continuous components.
* Some robustness against local minima, although hard to control.
* Great flexibility in parameterisation: mutation schemes, crossover schemes, fitness functions, selection functions, etc.

#### Cons

* Many, many "hyperparameters" to tune which radically affect the performance of the optimisation. How should you choose them?
* No guarantee of convergence; *ad hoc*.
* \(Very\) slow compared to using stronger knowledge of the objective function.
* Many evaluations of objective function are required: one per population member per iteration.

### Memory

The optimisation algorithms we have seen so far are **memoryless**. They investigate some part of the solution space, check the loss, then move on. They may end up checking the same, or very similar, solutions over and over again. This inefficiency can be mitigated using some form of **memory**, where the optimiser remembers where "good" and "bad" bits of the parameters space are, and  makes decisions using this memory. In particular, we want to remember good **paths in solution space**. 

#### Pros
* Can be very effective in spaces where good solutions are separated by large, narrow valleys.
* Can use fewer evaluations of the objective function than genetic algorithm if pheromones are effective.
* When it works, it really works.

#### Cons
* Moderately complex algorithm to implement.
* No guarantee of convergence; *ad hoc*.
* Even *more* hyperparameters than genetic algorithms.
* People think you work with ants.

# Quality of optimisation

### Convergence	

An optimisation algorithm is said to **converge** to a **solution**. In convex optimisation, this means that the **global minimum** has been found and the problem is solved. In non\-convex optimisation, this means a **local minimum** has been found from which the algorithm cannot escape.

A good optimisation algorithm converges quickly. This means that the drop in the objective function should be steep, so that each iteration is making a big difference. A bad optimisation algorithm does not converge at all \(it may wander forever, or diverge to infinity\). Many optimisation algorithms only converge under certain conditions; the convergence depends on the initial conditions of the optimisation.

#### Guarantees of convergence

Some optimisation algorithms are guaranteed to converge if a solution exists; while others \(like most heuristic optimisation algorithms\) are not guaranteed to converge even if the problem has a solution. For example, a random search might wander the space of possibilities forever, never finding the specific configuration that minimises \(or even reduces\) the loss.

For iterative solutions, a plot of the objective function value against iteration is a helpful tool in diagnosing convergence problems. Ideally, the loss should drop as fast as possible.


### Tuning optimisation

Optimisation turns specific problems into ones that can be solved with a general algorithm, as long as we can write down an objective function. However, optimisation algorithms have **hyperparameters**, which affect the way in which the search for the optimum value is carried out. Using optimisers effectively requires adjusting these hyperparameters.

#### Use the right algorithm

* If you know the problem is **least-squares** use a specialised least-squares solver. You might be able to solve directly, for example with the pseudo\-inverse.
* If you know the problem is **convex**, use a convex solver. This is radically more efficient than any other choice if its applicable.
* If you know the derivatives of the objective function, or can compute them using automatic differentiation, use a **first-order** method \(or second order, if you can\)
* If you don't know any of these things, use a general purpose **zeroth\-order** solver like **simulated annealing** or a **genetic algorithm**.


### What can go wrong?
#### Slow progress

Slow progress typically occurs in local search where the steps made are too small. For example, gradient descent with a very small 𝛿 or hill climbing with a tiny neighbourhood function will only be able to search a very small portion of the space. This will correspond to a very slowly decreasing loss plot.

#### Noisy and diverging performance

Local search can also become unstable, particularly if jumps or steps are too large an the optimiser bounces around hopelessly. The optimisation can diverge if the objective function has infinitely decreasing values in some direction \("the abyss"\), and this typically requires constraints to limit the **feasible set**.


### Getting stuck

Some optimisers can get stuck, usually at critical points of the objective function.  
* **Plateaus** can cause memoryless algorithms to wander, and derivative\-based algorithms to cease moving entirely. Techniques like **momentum** and other forms of **memory** can limit this effect.
* **Local minima** can completely trap pure local search methods and halt progress. Some metaheuristics, like random restart can mitigate this.
* **Saddle points** can trap or slow gradient descent methods, which have trouble finding the *best* direction to go in when the function is increasing in some directions and decreasing in others. 
* **Very steep or discontinuous** objective functions can produce insurmountable barriers for gradient descent. Stochastic methods, like stochastic gradient descent, can "blur out" these boundaries and still make progress.

## Week 7 

**Deep learning** or **deep neural networks** have become a major part of modern machine learning research. They have had astonishing success in fields like speech recognition, machine translation, image classification and image synthesis. The basic problem of deep learning is one of finding an approximating function. In a simple model, this might work as follows: Given some observations x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub> and some corresponding outputs y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>n</sub>, find a function y' = f\(x;θ\) with parameters θ, such that we have:

θ' = min\(∑\|\|f\(xi;θ\)\|\|\)

where distance is some measure of how close the output of f and the expected output y<sub>i</sub> are \(the specific distance used varies with the problem\). The idea is that we can learn f such that we can generalise the transform to x we haven't seen yet. This is obviously an optimisation problem. But deep neural networks can have *tens of millions* of parameters - a very long θ vector. How is it possible to do optimisation in reasonable time in such a large parameter space?

### Backpropagation

The answer is that these networks are constructed in a very simple \(but clever way\). A traditional neural network consists of a series of **layers**, each of which is a **linear map** \(just a matrix multiply\) followed by a simple, fixed nonlinear function.  Think: rotate, stretch \(linear map\) and fold \(simple *fixed* nonlinear folding\). The output of one layer is the input for the next.

The linear map in each layer is \(of course\) specified by a matrix, known as the **weight matrix**. The network is completely parameterised by the entries of the weight matrices for each layer \(all of the entries of these matrices can be seen as the parameter vector θ\). The nonlinear function G\(x\) is  fixed for every layer and cannot vary; it is often a simple function which "squashes" the range of the output in some way \(like `tanh`, `relu` or `sigmoid` -- you don't need to know these\). Only the weight matrices change during optimisation.

y<sub>i</sub> = G\(W<sub>i</sub>x<sub>i</sub> \+ b<sub>i</sub>\)

### Why not use heuristic search?

Heuristic search methods like random search, simulated annealing and genetic algorithms are easy to understand, easy to implement and have few restrictions on the problems they can be applied to. So why not always use those approaches?

* They can be very slow; it may take many iterations to approach a minimum and require significant computation to compute each iteration.
* There is no guarantee of convergence, or even of progress. The search can get stuck, or drift slowly over plateaus.
* There are a huge number of **hyperparameters** that can be tweaked \(temperature schedules, size of population, memory structure, etc.\). How should these be chosen? Optimal choice of these parameters becomes an optimisation problem in itself.

For optimisation problems like deep neural networks, heuristic search is hopelessly inadequate. It is simply too slow to make progress in training networks with millions of parameters. Instead, **first-order** optimisation is applied. **First-order algorithms**, that we will discuss today, can be *orders of magnitude* faster than heuristic search.

*Note: if an objective function is known to be convex with convex constraints, there are much better methods still for optimisation, which can often run exceptionally quickly with guaranteed convergence.*

### Jacobian: matrix of derivatives

* If f\(x\) is a scalar function of a scalar x, f'\(x\) is the first derivative of f w.r.t. x, i.e. d / dx of f\(x\). The second derivative is written f''\(x\) = d<sub>2</sub> / dx<sub>2</sub> of f\(x\).

* If we generalise this to a **vector** function y = f\(x\), then we have a rate of change \(derivative\) between every component of the input and every component of an output at any specific input x. We can collect this derivative information into a matrix called the **Jacobian matrix**, which characterises the slope *at a specific point x*. If the input x ∈ n and the output y ∈ m, then we have an m x n matrix.

This simply tells us how much each component of the output changes as we change any component of the input -- the generalised "slope" of a vector\-valued function. This is a very important way of characterising the variation of a vector function at a point x, and is widely used in many contexts. In the case where f maps n → n \(from a vector space to the same vector space\), then we have a square n x n matrix J with which can we do standard things like compute the determinant, take the eigendecomposition or \(in some cases invert\).

In many cases, we have a very simple Jacobian: just one single row. This applies in cases where we have a scalar function y = f\(x\), where y \(i.e. a one dimensional output from an n dimensional input\). This is the situation we have with a loss function L\(θ\) is a scalar function of a vector input. In this case, we have a single row Jacobian: the gradient vector.

### Gradient vector: one row of the Jacobian

* ∇ f\(x\) is the *gradient vector* of a \(scalar\) function of a vector, the equivalent of the first derivative for vector functions. We have one \(partial\) derivative per component of x. This tells us how much f\(x\) would vary if we made tiny changes to each dimension *independently*. Note that in this course we only deal with function f\(x\) with scalar outputs, but with vector inputs. We will work with scalar objective functions L\(θ\) of parameter vectors θ.

∇ L\(θ\) is a vector *which points in the direction of the steepest change in L\(θ\)*.

### Hessian: Jacobian of the gradient vector

* ∇<sup>2</sup> f\(x\) is the *Hessian matrix* of a \(scalar\) function of a vector, the equivalent of the second derivative for vector functions.  

Following our rule above, it's just the Jacobian of a vector valued function, so we know:

* ∇<sup>2</sup> L\(θ\) is matrix valued map n → n x n

This is important, because we can see that the second derivative even of a scalar valued function scales quadratically with the dimension of its input!

\(if the original function was a vector, we'd have a Hessian tensor instead\).

### Differentiable objective functions 

For some objective functions, we can compute the (exact) **derivatives** of the  objective function with respect to the parameters θ. For example, if the objective function has a single scalar parameter θ ∈ 1 dim and the function is:  
L\(θ\) = θ<sup>2</sup>
then, from basic calculus, the derivative with respect to θ is just:  
L'\(θ)\ = 2θ

If we know the derivative, we can use this to move in "good directions" -- down the slope of the objective function towards a minimum.

This becomes slightly more involved for multidimensional objective functions \(where θ has more than one component\) where we have a **gradient vector** instead of a simple scalar derivative \(written ∇ L\(θ\)\). However, the same principle applies.

#### Orders: zeroth, first, second

Iterative algorithms can be classified according to the order of derivative they require:
* a **zeroth order**  optimisation algorithm only requires evaluation of the objective function L\(θ\). Examples include random search and simulated annealing.

* a **first order**  optimisation algorithm requires evaluation of L\(θ\) and its derivative ∇ L\(θ\). This class includes the family of **gradient descent** methods.

* a **second order** optimisation algorithm requires evaluation L\(θ\), ∇ L\(θ\) and ∇ ∇ L\(θ\)$. These methods include **quasi\-Newtonian** optimisation.


### Optimisation with derivatives

If we know \(or can compute\) the **gradient** an objective function, we know the **slope** of the function at any given point. This gives us both:  
* the direction of fastest increase and
* the steepness of that slope. 

This is  *the* major application of calculus. 

Knowing the derivative of the objective function is sufficient to dramatically improve the efficiency of optimisation. 

### Conditions

### Differentiability

A **smooth function** has continuous derivatives up to some order. Smoother functions are typically easier to do iterative optimisation on, because small changes in the current approximation are likely to lead to small changes in the objective function. We say a function is *C<sup>n</sup> continuous* if the nth derivative is continuous.

There is a difference between *having continuous derivatives* and *knowing what those derivatives are*. 

First order optimisation uses the \(first\) **derivatives of the objective function with respect to the parameters**. These techniques can only be applied if the objective function is:
* At least *C<sup>1</sup> continuous* i.e. no step changes anywhere in the function or its derivative
* **differentiable** i.e. gradient is defined everywhere

\(though we will see that these constraints can be relaxed a bit in practice\).

Many objective functions satisfy these conditions, and first\-order methods can be vastly more efficient than zeroth\-order methods. For particular classes of functions \(e.g. convex\) there are known bounds on number of steps required to converge for specific first\-order optimizers.

### Lipschitz continuity \(no ankle breaking\)

First\-order \(and higher\-order\) continuous optimisation algorithms put a stronger requirement on functions than just $C^1$ continuity and require the function to be **Lipschitz continuous**.

For functions n → 1 dim \(i.e. the objective functions L\(θ\) we are concerned with\), this is equivalent to saying that the *gradient is bounded* and the function cannot change more quickly than some constant; there is a maximum steepness.  L(θ)/di ≤ K for all i and some fixed K.

### Lipschitz constant

We can imagine running a cone of a particular steepness across a surface. We can check if it ever touches the surface. This is a measure of how steep the function is; or equivalently the bound of the first derivative. The **Lipschitz constant** K of a function f\(x\) is a measure of wide this cone that only touches the function once is. This is a measure of how smooth the function is, or equivalently the maximum steepness of the objective function at any point anywhere over its domain. It can be defined as:

K = sup \[\|f\(x\)\-f\(y\)\| / \|x\-y\|\]

where sup is the supremum; the smallest value that is larger than every value of this function.

A smaller K means a function that is smoother. K=0 is totally flat. We will assume that the functions we will deal with have some finite K, though its value may not be precisely known.

### Analytical derivatives

If we have **analytical derivatives** \(i.e. we know the derivative of the function in closed form; we can write down the maths directly\), you will probably remember the "high school" process for mathematical optimisation:
* compute the derivative f'\(x\) = d/dx of f(x)
* solve for the derivative being zero \(i.e. solve x for f'\(x\)=0\). This finds all **turning points** or **optima** of the function.
* then check if any of the solutions has positive second derivative f'''\(x\)\>0, which indicates the solution is a minimum.

### Computable exact derivatives

The analytical derivative approach doesn't require any iteration at all. We get the solution immediately on solving for the derivative. But usually we don't have a simple solution for the derivative, but we can evaluate the derivative *at a specific point*; we have **exact pointwise derivatives**. We can evaluate the function f'\(x\) for any x but not write it down in closed form. In this case, we can still dramatically accelerate optimisation by taking steps such that we "run downhill" as fast as possible. This requires that we can compute the gradient at any point on the objective function. 

### Gradient: A derivative vector

We will work with objective functions that take in a vector and output a scalar:

```
    # vector -> scalar
    def objective(theta): 
        ...
        return score
```

We want to be able to *generate* functions:

``` 
    # vector -> vector
    def grad_objective(theta):
        ...
    return score_gradient
```    
This vector ∇ L\(θ\) is called the **gradient** or **gradient vector**. At any given point, the gradient of a function points in the direction where the function *increases fastest*. The magnitude of this vector is the rate at which the function is changing \("the steepness"\).

### Gradient descent

The basic first-order algorithm is called **gradient descent** and it is very simple, starting from some initial guess θ<sup>\(0\)</sup>:
    
θ<sup>\(i\+1\)</sup> = θ<sup>\(i\)</sup> \- δ ∇ L\(θ<sup>\(i\)</sup>\)      

where δ is a scaling hyperparameter -- the **step size**. The **step size** might be fixed, or might be chosen adaptively according to an algorithm like *line search*.

This means is that the optimiser will make moves where the objective function drops most quickly.

In simpler terms:

* starting somewhere θ<sup>\(0\)</sup>
* repeat:
    * check how steep the ground is in each direction v = ∇ L\(θ<sup>\(i\)</sup>\)
    * move a little step δ in the steepest direction v to find θ<sup>\(i\+1\)</sup>.

Notation note: θ<sup>\(i\)</sup> does not mean the ith power of θ but just the ith θ in a sequence of iterations: θ<sup>\(0\)</sup>, θ<sup>\(1\)</sup>, θ<sup>\(2\)</sup>, ...

### Downhill is not always the shortest route

While gradient descent *can* be very fast, following the gradient is not necessarily the fastest route to a minimum. In the example below, the route from the red point to the minimum is very short. Following the gradient, however, takes a very circuitous path to the minimum.

It is generally much faster than blindly bouncing around hoping to get to the bottom, though!

### Why step size matters

The step size ∇ is critical for success. If it is too small, the steps will be very small and convergence will be slow. If the steps are too large, the behaviour of the optimisation can become quite unpredictable. This happens if the gradient functions changes significantly over the space of a step \(e.g. if the gradient changes sign across the step\).

### Relationship to Lipschitz constant

We won't show this, but the optimal step size δ is directly related to the Lipschitz constant K of the objective function. Unfortunately we rarely know K precisely in many real\-world optimisation tasks, and step size is often set by approximate methods like line search.

### Gradient descent in 2D

This technique extends to any number of dimensions, as long as we can get the gradient vector at any point in the parameter space. We just have a gradient vector ∇ L\(θ\) instead of a simple 1D derivative. There is no change to the code.

### Gradients of the objective function

For first\-order optimisation to be possible, the derivative of the objective function has to be available. This obviously does not apply directly to empirical optimisation \(e.g. real world manufacturing where the quality of components is being tested in an experiment -- there are no derivatives\), but it can be applied in many cases where we have a computational model that can be optimised. This is again a reason to favour building models when optimising.

### Why not use numerical differences?
The definition of differentiation of a function f\(x\) is the well known formula:

d/dx of f\(x\) = lim \[f\(x\+h\) \- f\(x\-h\)/2h\] for h to θ

Given this definition, why do we need to know the *true derivative* ∇ L\(θ\) if we can evaluate L\(θ\) anywhere? Why not just evaluate L\(θ\+h\) and L\(θ\-h\) for some small h? This approach is called numerical differentiation, and these are **finite differences**. 

This works fine for reasonably smooth one\-dimensional functions.

### Numerical problems

It is also difficult to choose h such that the function is not misrepresented by an excessive value but numerical issues do not dominate the result. Remember that finite differences violates *all* of the rules for good floating point results:
f\(x\+h\) \- f\(x\-h\)/2h

* \(a\) it adds a small number h to a potentially much larger number x *\(magnitude error\)*
* \(b\) it then subtracts two very similar numbers f\(x\+h\) and f\(x\-h\)  *\(cancellation error\)*
* \(c\) then it divides the result by a very small number 2h *\(division magnification\)*

It would be hard to think of a simple example that has more potential numerical problems than finite differences!

### Revenge of the curse of dimensionality

This is not useful in high dimensions, even if we can deal with numerical issues, however. The **curse of dimensionality** strikes once again. To evaluate the *gradient* at a point x we need to compute the numerical differences in *each* dimension x<sub>i</sub>. If θ has one million dimensions, then *each* individual derivative evaluation will require two million evaluations of L\(θ\)! This is a completely unreasonable overhead. The acceleration of first\-order methods over zeroth\-order would be drowned out by the evaluation of the gradient.

### Improving gradient descent

Gradient descent can be very efficient, and often **much** better than zeroth\-order methods. However, it has drawbacks:

* The gradient of the loss function L'\(θ\) = ∇ L\(θ\) must be computable at any point θ. **Automatic differentation** helps with this.

* Gradient descent can get stuck in local minima. This is an inherent aspect of gradient descent methods, which will not find global minima except when the the function is convex *and* the step size is optimal. **Random restart** and **momentum** are approachs to reduce sensitivity to local minima.
* Gradient descent only works on smooth, differentiable objective functions. **Stochastic relaxation** introduces randomness to allow very steep functions to be optimised.
* Gradient descent can be very slow if the objective function \(and/or the gradient\) is slow to evaluate. **Stochastic gradient descent** can massively accelerate optimisation if the objective function can be written as a simple sum of many subproblems.

### Automatic differentiation

This problem can be solved if we know analytically the derivative of the objective function in closed form. For example, the derivative of the least squares linear regression that we saw in the last lecture is \(relatively\) easy to work out exactly as a formula. However, it seems very constraining to have to manually work out the derivative of the objective function, which might be very complicated indeed for a complex multidimensional problem. This is the major motivation for **algorithmic differentiation** \(or **automatic differentiation**\).

Automatic differentiation can take a function, usually written a *subset* of a full programming language, and automatically construct a function that evaluates the exact derivative at any given point. This makes it feasible to perform first\-order optimisation.

### Autograd

*autograd* has now evolved into **Google JAX**, probably the most promising machine learning library. JAX supports GPU and TPU computation with automatic differentation. 

### Derivative tricks

\[*Side note: if you have the gradient of a function, you can do nifty things like computing the light that would be reflected from the surface, given a fixed light source. This isn't directly important for DF\(H\), but shows how useful being able to differentiate functions is*\]

### Using autograd in optimisation

Using automatic differentiation, we can write down the form of the objective function as a straightforward computation, and get the derivatives of the function "for free". This makes it extremely efficient to perform first\-order optimisation. 

This is what machine learning libraries do. They just make it easy to write vectorised, differentiable code that runs on GPU/TPU hardware. The rest is just dressing.

### Fitting a line, first\-order

Let's re\-solve the line of best fit from lecture 6. We want to find m and c; the parameters of a line, such that the square difference between the line and a set of datapoints is minimised \(the objective function\).

### Limits of automatic differentiation

Obviously, differentiation is only available for functions that are differentiable.  While first\-order gradient vectors are often computable in reasonable time, as we'll see later it becomes very difficult to compute second derivatives of multidimensional functions. 

### Stochastic relaxation

How do animals evolve camouflage? This question is posed and discussed in **"The Blind Watchmaker"** by *Richard Dawkins*. Evolution is a gradual optimisation process, and to make steps that might be accepted requires that there is a smooth path from "poor fitness" to "good fitness".

### Resolution

The argument is that although every *specific* case is a simple binary choice, it is *averaged* over many random instances, where the conditions will be slightly different \(maybe it is nearly dark, maybe the predator has bad eyes, maybe the weather is foggy\) and averaging over all of those cases, some very minor change in colouring might offer an advantage. Mathematically speaking, this is **stochastic relaxation**; the apparently impossibly steep gradient has been rendered \(approximately\) Lipschitz continuous by integrating over many different random conditions.

This is applicable to many problems outside of evolution. For example, a very steep function has a very large derivative at some point; or it might have zero derivative in parts. But if we average over lots of cases where the step position is very slightly shifted, we get a smooth function.


### Stochastic gradient descent

Gradient descent evaluates the objective function and its gradient at each iteration before making a step. This can be very expensive to do, particularly when optimising function approximations with large data sets (e.g. in machine learning). 

If the objective function can be broken down into small parts, the optimiser can do gradient descent on randomly selected parts independently, which may be much faster. This is called **stochastic gradient descent \(SGD\)**, because the steps it takes depend on the random selection of the parts of the objective function. 

This works if the objective function can be written as a sum:

L\(θ\) = ∑ L<sub>i</sub>\(θ\)

i.e. that the objective function is composed of the sum of many simple sub-objective functions L<sub>1</sub>\(θ\), L<sub>2</sub>\(θ\), ..., L<sub>n</sub>\(θ\).

This type of form often occurs when matching parameters to observations -- **approximation problems**, as in machine learning applications. In these cases, we have many **training examples** x<sub>i</sub> with matching known outputs y<sub>i</sub>, and we want find a parameter vector θ such that:

L\(θ\) = \∑ \|\|f\(xi;θ\) \- y<sub>i</sup> \|\|

is minimised, i.e. that the difference between the model output and the expected output is minimised, *summing over all training examples*. 

Differentiation is a linear operator. This means that we can interchange summation, scalar multiplication and differentiation 
d/dx\(a f\(x\) \+ b g\(x\)\) = a d/dx of f\(x\) + b d/dx of g\(x\), we have:

∇ \∑ \|\|f\(xi;θ\) \- y<sub>i</sup> \|\| = ∑ ∇ \|\|f\(xi;θ\) \- y<sub>i</sup> \|\|

In this case, we can take any *subset* of training samples and outputs, compute the gradient for each sample, then make a move according to the computed gradient of the subset. Over time, the random subset selection will \(hopefully\) average out. Each subset is called a **minibatch** and one run through the whole dataset \(i.e. enough minibatches so that every data item has been "seen" by the optimiser\) is called an **epoch**.

### Linear regression with SGD

We can do our linear regression example with 10000 points - and find a good fit with *only one pass through the data*. This works because we can divide the problem up into a lots of sums of smaller problems \(fitting a line on a few random points at a time\) which are all part of one big problem \(fitting a line to all of the points\). 

When this is possible, it is vastly more efficient than computing the gradient for the entire set of data we have available.

### Random restart

Gradient descent gets trapped in minima easily. Once it is in an **attractor basin** of a **local minima** it can't easily get out. SGD can add a little bit of noise that might push the optimiser over small ridges and peaks, but not out of deep minima. A simple heuristic is just to run gradient descent until it gets stuck, then randomly restart with different initial conditions and try again. This is repeated a number of times, hopefully with one of the optimisation paths ending up in the global minimum. This metaheuristic works for any local search method, including hill climbing, simulated annealling, etc.

### Simple memory: momentum terms

A physical ball rolling on a surface is not stopped by a small irregularity in a surface. Once it starts rolling downhill, it can skip over little bumps and divots, and cross flat plains, and head steadily down narrow valleys without bouncing off the edges. This is because it has **momentum**; it will tend to keep going in the direction it was just going. This is a form of the memory metaheuristic. Rather than having ant-colony style paths everywhere, the optimiser just remembers one simple path -- the way it is currently going.

The same idea can be used to reduce the chance of \(stochastic\) gradient descent becoming trapped by small fluctuations in the objective function and "smooth" out the descent. The idea is simple; if you are going the right way now, keep going that way even if the gradient is not always quite downhill.

We introduce a velocity v, and have the optimiser move in this direction. We gradually adjust v to align with the derivative.

v = α v \+ δ ∇ L\(θ\)  
θ<sup>\(i\+1\)</sup> = θ<sup>\(i\)</sup> \- v


This is governed by a parameter α; the closer to α is to 1.0 the more momentum there is in the system. α=0.0 is equivalent to ordinary gradient descent.

### Types of critical points

This physical surface intuition leads us to think of how we would characterise different parts of an objective function. 

For smooth, continuous objective functions, there are various points in space of particular interest. These are **critical points**, where the gradient vector components is the zero vector.

### Second\-order derivatives

If the first order derivatives represent the "slope" of a function, the second order derivatives represent the "curvature" of a function.

For every parameter component θ<sub>i</sub> the Hessian stores how the *steepness* of every other θ<sub>j</sub> changes.

### Eigenvalues of the Hessian

The Hessian matrix captures important properties about the **type of critical point** that we saw in the previous lecture. In particular, the **eigenvalues** of the Hessian tell us what kind of point we have.

* If all eigenvalues are all strictly positive, the matrix is called **positive definite** and the point is a minimum.
* If all eigenvalues are all strictly negative \(**negative definite**\) and the point is a maximum.
* If eigenvalues have mixed sign the point is a saddle point.
* If the eigenvalues are all positive/negative, but with some zeros, the matrix is **semidefinite** and the point is plateau/ridge.

### Second\-order optimsation

Second\-order optimisation uses the Hessian matrix to jump to the bottom of each local quadratic approximation in a single step.  This can skip over flat plains and escape from saddle points that slow down gradient descent. Second\-order methods are *much* faster in general than first\-order methods. 

#### Curse of dimensionality \(yet again\)

However, simple second order methods don't work in high dimensions. Evaluating the Hessian matrix requires d<sup>2</sup> computations, and d<sup>2</sup> storage. Many machine learning applications have models with d \> 1 million parameters. Just storing the Hessian matrix for *one iteration* of the optimisation would require:

## Week 8

### What is probability?

* **Experiment** \(or **trial**\) An occurrence with an uncertain outcome.
    * For example, losing a submarine -- the location of the submarine is now unknown.
* **Outcome** The result of an experiment; one particular state of the world. 
    * For example: the submarine is in ocean grid square \[2,3\].
* **Sample Space** The set of *all possible* outcomes for an experiment. 
    * For example, ocean grid squares \{\[0,0\], \[0,1\], \[0,2\], \[0,3\], ..., \[8,7\], \[8,8\], \[8,9\], \[9,9\]\}.
* **Event** A *subset* of possible outcomes with some common property. 
    * For example, the grid squares which are south of the Equator.
* **Probability** The probability of an event *with respect to a sample space* is the number of outcomes from the sample space that are in the event, divided by the total number of outcomes in the sample space. Since it is a ratio, probability will always be a real number between 0 \(representing an impossible event\) and 1 \(representing a certain event\).  
    * For example, the probability of the submarine being below the equator, or the probability of the submarine being in grid square \[0,0\] \(in this case the event is just a single outcome\).
* **Probability distribution** A mapping of outcomes to probabilities that sum to 1. This is because an outcome must happen from a trial (with probability 1) so the sum of all possible outcomes together will be 1. A random variable has a probability distribution which maps each outcome to a probability.
    * For example P\(X=x\), the probability that the submarine is in a specific grid square x.
* **Random variable** A variable representing an unknown value, whose probability distribution we *do* know. The variable is associated outcomes of a trial    
    * For example, X is a random variable representing the location of the submarine. 
* **Probability density/mass function** A function that *defines* a probability distribution by mapping each outcome to a probability f<sub>X</sub>\(x\), x → 1 dim. This could be a continuous function over x \(density\) or discrete function over x \(mass\).
    * For example f<sub>X</sub>\(x\) would be a probability mass function for the submarine, which maps each grid square to real number representing its probability.
* **Observation** An outcome that we have directly observed; i.e. data.
    * For example, a submarine was found in grid square \[0,5\]
* **Sample** An outcome that we have simulated according a probability distribution. We say we have **drawn** a sample from a distribution.
    * For example, if we believe that the submarine was distributed according to some pattern, generate possible concrete grid positions that follow this pattern.
* **Expectation/expected value** The "average" value of a random variable.
    * The submarine was on average in grid square \[3.46, 2.19\]
    
### Superiority of probabilistic models

Regardless of the philosophical model you subscribe to, there is one thing you can be sure of: *probability is the best*. 

There are other models of uncertainty than probability theory that are sometimes used. However, all other representations of uncertainty are *strictly inferior* to probabilistic methods *in the sense that* a person, agent, computer placing "bets" on future events using probabilistic models has the best possible return out of all decision systems when there is uncertainty. 

> *Any theory with as good a gambling outcome as would be achieved using probability theory is equivalent to probability theory.*

----

### Generative models: forward and inverse probability

A key idea in probabilistic models is that of a **generative process**; the idea that there is some unknown process going on, the results of which can be observed. The process itself is governed by unobserved variables that we do not know but which we can **infer**.

The classic example is an **urn problem**. Consider an urn, into which a number of balls have been poured \(by some mysterious entity, say\). Each ball can be either black or white. 

You pull out four random balls from the urn and observe their colour. You get four white balls. 

There are lots of questions you can ask now:

* What is the probability that the next ball that is drawn will be white? 
    * This is a **forward probability** question. It asks questions related to the distribution of the observations.
* What is the distribution of white and black balls in the urn? 
    * This is an **inverse probability** question. It asks questions related to unobserved variables that govern the process that generated the observations.
* Who is the mysterious entity?
    * This is an unknowable question. The observations we make cannot resolve this question.
    
There are a huge number of processes that can be framed as urn problems \(urns where balls are replaced, problems where there are multiple urns and you don't know which urn the balls came from, problems where balls can move between urns, and so on\). 

### A formal basis for probability theory
#### Axioms of probability

There are only a few basic axioms of probability, from which everything else can be derived. Writing P\(A\) to mean the probability of event A \(NOTE: these apply to **events** \(sets of outcomes\), not just outcomes!\):
    
* **Boundedness**
0 ⩽ P\(A\) ⩽ 1  
 all possible events A -- probabilities are 0, or positive and less than 1.  
* **Unitarity**
Σ P\(A\)=1  
for the complete set of possible **outcomes** \(not events!\) A ∈ σ in a sample space σ -- something always happens.
* **Sum rule**
P\(A ∨ B\) = P\(A\) + P\(B\) \- P\(A ∧ B\),
i.e. the probability of either event A or B happening is the sum of the independent probabilities minus the probability of both happening.  
* **Conditional probability**
The conditional probability P\(A\|B\) is defined to be the probability that event A will happen *given that we already know B to have happened*.
P\(A\|B\) = P\(A ∧ B\) \ P\(B\)

### Random variables and distributions

A **random variable** is a variable that can take on different values, but we do not know what value it has; i.e. one that is "unassigned". However, we have some knowledge which captures the possible states the variable could take on, and their corresponding probabilities. Probability theory allows us to manipulate random variables without having to assign them a specific value.

A random variable is written with a capital letter, like X.

A random variable might represent:

* the outcome of dice throw \(discrete\); 
* whether or not it is raining outside \(discrete: binary\); 
* the latitude of the USS Scorpion \(continuous\); 
* the height of person we haven't met yet \(continuous\). 

### Distributions

A **probability distribution** defines how likely different states of a random variable are. 

We can see X as the the *experiment* and x as the *outcome*, with a function mapping every possible outcome to a probability. We write P\(X=x\) \(note the case!\), and use the shorthand notations:

P\(X=x\), the probability of random variable X taking on value x  
P\(X\), shorthand for probability of X=x   
P\(x\), shorthand for probability of specific value X=x   

We can see an outcome as a random variable taking on a specific value i.e. P\(X=x\). Note that by convention we use P\(A\) to mean the probability of **event** A, not a random variable A \(an **event** is a *set* of **outcomes**; **random variables** only assign probabilities to **outcomes**\).

#### Discrete and continuous

Random variables can be continuous \(e.g. the height of a person\) or discrete \(the value showing on the face of a dice\). 

* **Discrete variables** The distribution of a discrete random variable is described with a **probability mass function** \(PMF\) which gives each outcome a specific value; imagine a Python dictionary mapping outcomes to probabilities. The PMF is usually written f<sub>X</sub>\(x\), where P\(X=x\) = f<sub>X</sub>\(x\).

* **Continuous variables** A continuous variable has a **probability density function** \(PDF\) which specifies the spread of the probability over outcomes as a *continuous function* f<sub>X</sub>\(x\). It is **not** the case that P\(X=x\) = f<sub>X</sub>\(x\) for PDFs.

#### Integration to unity

A probability mass function or probability density function *must* sum/integrate to exactly 1, as the random variable under consideration must take on *some* value; this is a consequence of unitarity. Every repetition of an experiment has exactly one outcome.

∑ f<sub>X</sub>\(x<sub>i</sub>\) = 1 for PMFs of discrete RVs
∫ f<sub>X</sub>\(x\) dx = 1 for PDFs of continuous RVs

# Expectation

If a random variable takes on numerical values, then we can define the **expectation** or **expected value** of a random variable X as:

X = ∫ f<sub>X</sub>\(x\) dx 

For a discrete random variable with probability mass function P\(X=x\) = f<sub>X</sub>\(x\), we would write this as a summation:

X = ∑ f<sub>X</sub>\(x\) x 

If there are only a finite number of possibilities, then this is: X = P\(X=x<sub>1</sub>) x<sub>1</sub> \+ P\(X=x<sub>2</sub>) x<sub>2</sub> \+ ... \+ P\(X=x<sub>n</sub>) x<sub>n</sub>

The expectation is the "average" of a random variable. Informally, it represents what we'd "expect to happen"; the most likely overall "score". It can be thought of as a *weighted sum* of all the possible outcomes of an experiment, where each outcome is weighted by the probability of that outcome occurring.  

### Expectation and means

Expectation corresponds to the idea of a **mean** or **average** result. The expected value of a random variable is the **true average** of the value of all outcomes that would be observed if we ran the experiment an infinite number of times. This is the **population mean** -- the mean of the whole, possibly infinite, population of a random variable.

Many important properties of random variables can be defined in terms of expectation. 

* The mean of a random variable X is just expected X. It is a measure of **central tendency**.
* The variance of a random variable X is X = \(X - expected X\)<sup>2</sup>. It is a measure of **spread**.

### Expectations of functions of X

We can apply functions to random variables, for example, the square of a random variable.

 The expectation of any function g\(X\) of a continuous random variable $X$ is defined as:  
expected g\(X\) = ∫ f<sub>X</sub>\(x\) g\(x\) dx
or  
expected g\(X\) = ∑ f<sub>X</sub>\(x\) g\(x\) dx  
for a discrete random variable.

We just take the sum/integral of each outcome, passed through the function g\(x\), weighted by the probability of the outcome x. g\(x\) could be thought of a "scoring" function, which assigns a real number to every outcome of X. Note that g\(x\) has no effect on the probability density/mass function f<sub>X</sub>\(x\). It doesn't affect the probability of the outcomes, just the *value assigned to those outcomes*.

Expected values are essential in making **rational decisions**, the central problem of **decision theory**. They combine scores \(or **utility**\) with uncertainty \(**probability**\).

### Samples and sampling

**Samples** are observed outcomes of an experiment; we will use the term **observations** synonymously, though samples usually refer to simulations and observations to concrete real data. 

We can **sample** from a distribution; this means simulating outcomes according to the probability distribution of those variables. We can also **observe** data which comes from an external source, that might believe is generated by some probability distribution.

### The empirical distribution

For discrete data, we can estimate the probability mass function that might be generating **observations** by 
counting each outcome seen divided by the total number of trials. This is called the **empirical distribution**.

This can be thought of as the **normalized histogram** of counts of occurrences of outcomes.

### Computing the empirical distribution 

For discrete random variables, we can always compute the empirical distribution from a series of observations; for example from the counts of a specific word  in a *corpus* of text \(e.g. in every newspaper article printed in 1994\). We just count the number of times each word is seen and divide by the total number of words.
    
P\(X=x\) = n<sub>x</sub> / N

where n<sub>x</sub> is the number of time outcome x was observed, and N is the total number of trials.


Note that the empirical distribution is a distribution which *approximates* an unknown true distribution. For very large samples of discrete variables, the empirical distribution will increasingly closely approximate the **true PMF**, assuming the samples we see are drawn in an unbiased way. However, this approach does not work usefully for continuous random variables, since we will only ever see each observed value once.

### Random sampling procedures

#### Uniform sampling

There are algorithms which can generate continuous random numbers which are **uniformly distributed** in an interval, such as from 0.0 to 1.0. These are actually **pseudo\-random numbers** in practice, since computers are \(hopefully\) deterministic. They designed to approximate the statistical properties of true random sequences. Inherently, all such generators generate sequences of discrete symbols \(bits or integers\) which are then mapped to floating point numbers in a specific range; this is quite tricky to get right.

> We must be careful: computers generate **pseudo\-random floating\-point numbers**; and not **true random real numbers**. While this makes little difference much of the time, they are quite different things.

A **uniformly distributed** number has equal probability of taking on any value in its interval, and zero probability every where else. Although this is sampling from a continuous PDF, it is the key building block in sampling from arbitrary PMFs. A uniform distribution is notated X ∼ U\(a,b\), meaning X is random variable which may take on values between a and b, with equal possibility of any number in that interval. The symbol ∼ is read "distributed as", i.e. "X is distributed as a uniform distribution in the interval \[a,b\]".

> Note that in practice these are not uniform across the reals in a given interval if we are using floating point, because we can only ever sample valid floating point values. While floats are non\-uniformly distributed, the difference isn't important for most applications.

### Writing and manipulation of probabilities 

Probabilities can be used to represent belief. But the raw numbers \(e.g. P\(X=x\) = 0.9999\) are not always a useful to make judgments, communicate results, or in some cases, even to do computations. 

### Odds, log odds

The **odds** of an event with probability $p$ is defined by:
    
odds =  1\-p / p

The odds are a more useful unit for discussing unlikely scenarios \(odds of 999:1 is easier to understand than p=0.001\).

**Log\-odds** or **logit** are particularly useful for very unlikely scenarios:

logit\(p\) = log \( p / 1\-p \)

### Log probabilities

The probability of multiple *independent* random variables taking on a set of values can be computed from the product:
P\(X,Y,Z\) = P\(X\)P\(Y\)P\(Z\)
and in general
    
P\(X<sub>1</sub>=x<sub>i</sub>, ..., X<sub>n</sub>=x<sub>n</sub>\) = product\(P\(X<sub>i</sub>=x<sub>i</sub>\)\) for i in range \(1, n\)

We often have to have to compute such products, but to multiply lots of values <1 leads to numerical issues: we will get floating point underflow. Instead, it is numerically more reliable to manipulate **log probabilities**, which can be summed instead of multiplied:

    
log \[P\(x<sub>1</sub>, ..., x<sub>n</sub>\)\] = ∑ log\(P\(x<sub>i</sub>\)\) for i in range \(1, n\)


This uses the identity log\(AB\) = log\(A\) + log\(B\)

This is simply a numerical convenience which avoids underflow. The **log-likelihood** is just log P\(B\|A\), and is often more convenient to work with than the raw likelihood.

### Comparing log-likelihoods

We could imagine that writing plays and novels is an activity that mysterious entities do by generating random characters according to a PMF. Under this \(very simplified!\) assumption, we could now take an "unknown" text \(in this case *Macbeth*\) and then look at how likely it would have been to have been generated under two models:
* *A* It was generated by a mysterious entity using the PMF for *Romeo and Juliet*
* *B* It was generated by a mysterious entity using the PMF for *Metamorphosis*

Neither of these will be exactly true, but we can precisely quantify to what extent *Macbeth* appears to have been generated by a similar process to these two reference texts.

This is a very rough proxy for whether or not they were generated by the same mysterious entity -- i.e. author. Our model is just the distribution of characters, so is a fairly weak model of different styles. However, it is sufficient to do the comparison here.

### Bayes' Rule

#### Prior, likelihood, posterior

##### Inverting conditional distributions

We often want to know the probability of a some event A given some other event B; that is P\(A\|B\). But we are often in the situation that we can only compute P\(B\|A\). 

This case is usually:
* we know how the mysterious entity behaves P\(B\|A\)
* we know what data we saw P\(B\)
* we know what the mysterious entity is likely to be up to in general;
* and we want to work out what the mysterious entity is doing P\(A\|B\).

In general P\(A\|B\) ≠ P\(B\|A\) and the two expressions can be completely different. 

Typically, this type of problem occurs where we:
* want to know the probability of some event given some *evidence* \(*how likely is it that I have a disease given that my blood test came back positive?*\) 
* but we only know the probability of observing evidence given the event \(*if you have this disease, the blood test will come back positive 95% of the time*\).

**Bayes' rule** gives the correct way to invert the probability distribution:
    
P\(A\|B\) = P\(B\|A\) P\(A\) / P\(B\)

This follows directly from the axioms of probability. Bayes' Rule is a very important rule, and has some surprising consequences. 

#### Nomenclature

* P\(A\|B\) is called the **posterior** -- what we want to know, or will know after the computation
* P\(B\|A\) is called the **likelihood** -- how likely the event A is to produce the evidence we see
* P\(A\) is the **prior**  -- how likely the event A is regardless of evidence
* P\(B\) is the **evidence** -- how likely the evidence B is regardless of the event.

Bayes' rule gives a consistent rule to take some prior belief and combine it with observed data to estimate a new distribution which combines them.

We often phrase this as some **hypothesis** $H$ we want to know, given some **data** $D$ we observe, and we write Bayes' Rule as:
    
P\(H\|D\) = P\(D\|H\) P\(H\) / P\(D\)

H and D are random variables in this expression.

\(the probability of the hypothesis given the data\) is equal to \(the probability of the data given the hypothesis\) times \(the probability of the hypothesis\) divided by \(the probability of the data\). In other words, if we want to work out how likely a hypothesis is to be true given observations, but we only know how likely we are to have seen those observations if that hypothesis *was* true, we can use Bayes' rule to solve the problem.

### Integration over the evidence

We can say that the posterior probability is *proportional* to the product of the prior and the likelihood. But to evaluate its value, we need to compute P\(D\), **the evidence**. 

It is difficult to see what this represents at first. But one way to think of it is as the result of marginalising the P\(D\) from the joint distribution P\(H,D\); that is integrating P\(H,D\) over every possible outcome of H, for each possible D. 

Because probabilities must add up to 1, we can write P\(B\) as:
P\(D\)  = Σ P\(D\|H<sub>i</sub>\) P\(H<sub>i</sub>\)
for a set of discrete outcomes  A<sub>i</sub> or
P\(D\) = ∫ P\(D\|H\) P\(H\) dA for a continuous distribution of outcomes.

**This trick is essential in understanding Bayes Rule!**

In general this can be difficult to compute. For binary simple cases where there are only two possible outcomes \(H can only be 0 or 1\), Bayes' rule can be written as:

P\(H=1\|D\) = P\(D\|H=1\)P\(H=1\) / P\(D\|H=1\)P\(H=1\) \+ P\(D\|H=0\) D\(H=0\)
    
### Natural frequency

There is an approach to explaining problems like this which makes it much less likely to make poor judgements. **Natural frequency** explanations involve imagining concrete populations of a fixed size \(10000 people in a room, for example\), and considering the proportions of the populations as *counts* \(how many people in the room have the plague?\).

### Bayes' rule for combining evidence

Bayes' rule is the correct way to combine prior belief and observation to update beliefs. We always transform from one probability distribution \(prior\) to a new belief \(posterior\) using some observed evidence. This can be used to "learn", where "learning" means updating a probability distribution based on observations. It has enormous applications anywhere uncertain information must be fused together, whether from multiple sources \(e.g. sensor fusion\) or over time \(e.g. probabilistic filtering\). 

### Entropy

A key property of a probability distribution is the **entropy**. Intuitively, this is a measure of the "surprise" an observer would have when observing draws from the distribution, or alternatively, the \(log\) measure of a number of distinct "states" a distribution could represent. A flat, uniform distribution is very "surprising" because the values are very hard to predict; a narrow, peaked distribution is unsurprising because the values are always very similar. 

This is a precise quantification -- it gives the *information* in a distribution. The units of information are normally bits; where 1 bit of information tells you the answer to exactly one yes or no question. The entropy tells you exactly how many of bits are needed \(at minimum\) to communicate a value from a distribution to an observer *who knows the distribution already*. Alternatively, you can see the number of distinct states the distribution describes as $p = 2<sup>H\(X\)</sup> -- this value is called the **perplexity**, and it can be fractional.

### Entropy is just the expectation of log\-probability

The entropy of a (discrete) distribution of a random variable X can be computed as:
    
H\(X\) = ∑ \-P\(X=x\) log<sub>2</sub>\(P\(X=x\)\)

This is just the expected value of the log\-probability of a random variable \(the "average" log\-probability\).

### Tossing coins

Consider a coin toss: this is sampling from a discrete random variable that can take on two states, heads and tails. 

If we call our two possible states 0 \(heads\) and 1 \(tails\), we can characterise this with a single parameter q, where P\(X=0\)=q and P\(X=1\)=\(1\-q\) \(this follows from the fact that as P\(X=0\)\+P\(X=1\) *must* equal 1 -- the coin must land on one side or the other. We ignore edge landings!\).

If this process is very biased and heads are much more likely than tails \(q<<0.5\), an observer will be unsurprised most of the time because predicting heads will be a good bet. If the process is unbiased \(q=0.5\), an observer will have no way to predict if a head or tail is more likely. We can write the entropy of this distribution:
H\(X\) = P\(X=0\) log<sub>2</sub> P\(X=0\) + P\(X=1\) log<sub>2</sub> P\(X=1\)
 = \-q log<sub>2</sub> q \- \(1\-q\) log<sub>2</sub> P\(1\-q\)

#### Interpreting entropy

We can see that observing a "q" means the next character isn't surprising at all: we know for *sure* that it will be a "u", and thus the entropy is 0.  There is only one configuration of the world for the next character \(under this model\) if we have just seen a "u".

Likewise, seeing a space character gives us very little information, and the next character could be anything: the character after a space will surprise us -- there are lots of configurations of the world that might follow a space. That surprise is as much surprise as we would get by tossing about 4 coins, or an entropy of about 4 bits.

## Week 9

### Continuous random variables
#### Problems with continuous variables

Continuous random variables are defined by a PDF \(probability *density* function\), rather than a PMF \(probability *mass* function\). A PMF essentially just a vector of values, but a PDF is a function mapping *any* input in its domain to a probability.  
This seems like a subtle difference\ (as a PMF has more and more "bins" it gets closer and closer to a PDF, right?\), but it has a number of complexities.

* The probability of any specific value is P\(X=x\)=0 *zero* for every possible x, yet any value in the *support* of the distribution function \(everywhere the PDF is non\-zero\) is possible. 

* There is no direct way to sample from the PDF in the same way as we did for the PMF. But there are several tricks for sampling from continuous distributions.

* We cannot estimate the true PDF from simple counts of observations like we did for the empirical distribution. This can never "fill in" the PDF, because it will only apply to a single value with zero "width".

* Bayes' Rule is easy to apply to discrete problems. But how do we do computations with continuous PDFs using Bayes' Rule?

* Simple discrete distributions don't have a concept of dimension. But we can have continuous values in 1 dim, or in vector spaces n, representing the probability of a random variable taking on a vector value, or indeed distributions over other generalisations \(matrices, other fields like complex numbers or quaternions and even more sophisticated structures like Riemannian manifolds\).

### Probability distribution functions

The PDF f<sub>X</sub>\(x\) of a random variable X maps a value x \(which might be a real number, or a vector, or any other continuous value\) to a single number, the density at the point. It is a function \(assuming a distribution over real vectors\) n → \+, where \+ is the positive real numbers, and ∫ f<sub>X</sub>\(x\)=1.

* While a PMF can have outcomes with a probability of at most 1, it is *not* the case that the maximum value of a PDF is f<sub>X</sub>\(x\) ≤ 1 -- *just that the integral of the PDF be 1.*

The value of the PDF at any point is **not** a probability, because the probability of a continuous random variable taking on any specific number must be zero. Instead, we can say that the probability of a continuous random variable X lying in a range \(a,b\) is:
    
P\(X ∈ \(a,b\)\) = \(a < X < b\)  = ∫ f<sub>X</sub>\(x\) from a to b

#### Support

The **support** of a PDF is the domain it maps from where the density is non-zero. 

supp\(x\) = x such that f<sub>X</sub>\(x\) > 0

Some PDFs have density over a fixed interval, and have zero density everywhere else. This is true of the uniform distribution, which has a fixed range where the density is constant, and is zero everywhere. This is called **compact support**. We know that sampling from a random variable with this PDF will always give us values in the range of this support. Some PDFs have non-zero density over an infinite domain. This is true of the normal distribution. A sample from a normal distribution could take on *any* real value at all; it's just much more likely to be close to the mean of the distribution than to be far away. This is **infinite support**.

### Cumulative distribution functions

The **cumulative distribution function** or CDF of a real\-valued random variable is 
    
F<sub>X</sub>\(x\) = ∫ f<sub>X</sub>\(x\) from \-∞ to x = P\(X ≤ x\)


Unlike the PDF, the CDF always maps $x$ to the codomain \[0,1\]. For any given value F<sub>X</sub>\(x\) tells us how much probability mass there is that is less than or equal to x. Given a CDF, we can now answer questions, like: what is the probability that random variable X takes on a value between 3.0 and 4.0?

P\(3 ≤ X ≤ 4\) = F<sub>X</sub>\(4\) \- F<sub>X</sub>\(3\)

This *is* a probability. Sometimes it is more convenient or efficient to do computations with the CDF than with the PDF.

### Location and scale

The normal distribution places the point of highest density at to its center μ \(the "mean"\), with a spread defined by σ<sup>2</sup> \(the "variance"\). This can be though of the **location** and **scale** of the density function. Most standard continuous random variable PDFs have a location \(where density is concentrated\) and scale \(how spread out the density is\).

#### Normal modelling

It seems that this might be a very limiting choice but there are two good reasons for this to work well as a model in many contexts:  
1. Normal variables have very nice mathematical properties and are easy to work with analyitically \(i.e. without relying on numerical computation\).
2. The *central limit theorem* tells us that any sum of random variables \(however they are distributed\) will tend to a *Levy stable distribution* as the number of variables being summed increases. For most random variables encountered, this means the normal distribution \(one specific Levy stable distribution\).

### Central limit theorem

If we form a sum of many random variables Y = X<sub>1</sub> \+ X<sub>2</sub> \+ X<sub>3</sub> \+ ..., then for almost any PDF that each of X<sub>1</sub>, X<sub>2</sub>,... might have, the PDF of Y will be approximately normal, Y ∼ N\(μ, σ\). This means that any process that involves a mixture of many random components will tend to be Gaussian under a wide variety of conditions.

### Multivariate distributions: distributions over n dims

Continuous distributions generalise discrete variables \(probability mass functions\) \(e.g. over Z\) to continuous spaces over 1 dim via probability density functions. 

Probability densities can be further generalised to vector spaces, particularly to n dims. This extends PDFs to assign probability across an entire vector space, under the constraint that the \(multidimensional integral\)     
∫ f<sub>X</sub>\(x) = 1 for x in n dims.  

Distributions with PDFs over vector spaces are called **multivariate distributions** \(which isn't a very good name; vector distributions might be clearer\). In many respects, they work the same as **univariate** continuous distributions. However, they typically require more parameters to specify their form, since they can vary over more dimensions.

#### Multivariate uniform

The multivariate uniform distribution is particularly simple to understand. It assigns equal density f<sub>X</sub>\(x<sub>i</sub>\) = f<sub>X</sub>\(x<sub>j</sub>\)$ to some \(axis\-aligned\) box in a vector space n dims, such that ∫ f<sub>X</sub>\(x\)=1, x ∈ a box. 

It is trivial to sample from; we just sample *independently* from a one-dimensional uniform distribution in the range [0,1] to get each element of our vector sample. This is a draw from a n\-dimensional uniform distribution in the unit box.

## Transformed uniform distribution

If we want to define a distribution over any box, we can simply transform the vectors with a matrix A and shift by adding an offset vector b.

### Normal distribution

The normal distribution \(above\) is very widely used as the distribution of continuous random variables. It can be defined for a random variable of *any dimension*; a **multivariate normal** in statistical terminology. In Unit 5, we saw the idea of a **mean vector** μ and a **covariance matrix** Σ which captured the "shape" of a dataset in terms of an ellipse. *These are in fact the parameterisation of the multivariate normal distribution.*

A multivariate normal is fully specified by a mean vector μ and the covariance matrix Σ. If you imagine the normal distribution to be a ball shaped mass in space, the mean *translates* the mass, and covariance applies a transformation matrix \(scale, rotate and shear\) to the ball.

Just like the uniform distribution, we can think of drawing samples from a "unit ball" with an independent normal distribution in each dimension. These samples are transformed linearly by the covariance matrix Σ and the mean vector μ, just like A and b above \(though Σ is actually $A<sup>\-1/2</sup> for technical reasons\)

### Joint and  marginal PDFs

We can look at the PDF of a multivariate normals for different covariances and mean vector \(centres and spreads\).

### Joint and marginal distributions

We can now talk about the **joint probability density function** \(density over all dimensions\) and the **marginal probability density function** \(density over some sub\-selection of dimensions\).

For example, consider $X ∼N\(μ, Σ\), X ∈ 2 dim, a two dimensional \("bivariate"\) normal distribution. We can look at some examples of the PDF, showing:

* Joint P\(X\)
* Marginal P\(X<sub>1</sub>\) and P\(X<sub>2</sub>\)
* Conditionals P\(X<sub>1</sub>\|X<sub>2</sub>\) and P\(X<sub>2</sub>\|X<sub>1</sub>\)

### Ellipses

When we spoke informally about the covariance matrix "covering" the data with an ellipsoidal shape, we more precisely meant that
*if* we represented the data as being generated with a normal distribution, and chose the mean vector and covariance matrix that best approximated that data, then the contours of the density of the PDF corresponding to equally spaced isocontours would be ellipsoidal.

### Monte Carlo 

How do we *draw* samples from a continuous distribution? How can we simulate the outcomes of a random variable X? This is a vital tool in computational statistics. One of the reasons computers are useful for statistical analysis is that they can generate \(pseudo\)\-random numbers very quickly.

#### von Neumann and Ulam

During the *Manhattan project* that developed the atomic bomb during the Second World War, there were many difficult probabilistic equations to work out. Although *analytical techniques* for solving certain kinds of problems existed, they were only effective some narrow types of problem and were tricky to apply to the problems that the Manhattan project had to solve.

John von Neumann and Stanislaw Ulam developed the **Monte Carlo** method to approximate the answer to probabilistic problems, named after the casinos of Monte Carlo. This involved setting up a *simulation* with stochastic \(random\) components. By running the simulation many times with different random behaviour, the population of *possible* behaviours could be approximated.

For example, computing the expectation of a function of a random variable can often be hard for continuous random variables. The integral for:

expected g\(X\) = ∫ f<sub>X</sub>\(x\) g\(x\) dx

may be intractable. However it is often very easy to compute g\(x\) for any possible x. If we can somehow sample from the distribution P\(X=x\), then we can approximate this very easily:
    
expected g\(X\) = ∫ f<sub>X</sub>\(x\) g\(x\) dx ≈ 1/N ∑ g\(x<sub>i</sub>\) for i in range\(1, N\)

where x<sub>i</sub> are random samples from P\(X=x\), defined by the PDF f<sub>X</sub>\(x\). This gets better as N gets larger.

### Throwing darts

For example, imagine trying to work out the expectation of dart throw. A dart board has sections giving different scores. We might model the position of the dart as a normal distribution over the dart space. This models the human variability in throwing. The expected score of a throw requires evaluating the integral of the normal PDF multiplied by the score at each point -- which isn't feasible to compute directly.


But we can sample from a multivariate normal distribution easily; we saw this in the last unit; just sample from d independent standard normals, and transform with a linear transform \(matrix\). So instead of trying to solve a very hard integral, we can simulate lots of dart throws, which follow the pattern of the normal distribution, and take the average score that they get. If we simulate a lot of darts, the average will be close to the true value of the integral.

### Inference

#### Population and samples, statistics and parameters

**Inferential statistics** is concerned with estimating the properties of an unobserved "global" **population** of values from a limited set of observed **samples**. This assumes that there is some underlying distribution from which samples are being drawn. This is a hidden process \(the "mysterious entity"\), which we only partially observe through the samples we see.

* **Population** is the unknown set of outcomes \(which might be infinite\)
    * **Example** the weight of all beetles
    * **Parameter** describes this **whole population**, e.g. the mean weight of all beetles
    
* **Sample** is some subset of the population that has been observed.
    * **Example** 20 beetles whose weight has been measured
    * **Statistic** is a function of the sample data, e.g. the arithmetic mean of those 20 samples
    
The parameters of the population distribution govern the generation of the samples that are observed. The problem of statistics is how to **infer** parameters given samples.

### Two worldviews

* **Bayesian inference** means that we consider *parameters* to be random variables that we want to refine a distribution over, and that data are fixed \(known, observed data\). We talk about belief in particular parameter settings.

* **Frequentist inference** means that we consider *parameters* to be fixed, but data to be random. We talk about how we approach an accurate estimate of the true parameters as our sample size increases.

### Three approaches

We will see three different approaches to doing inference:

* **Direct estimation** of parameters, where we define *functions of observations* that will estimate the values of parameters of distributions *directly*. This requires we assume the form of the distribution governing the mysterious entity. It is very efficient, but only works for very particular kinds of model. We need to have *estimator functions* for each specific distribution we want to estimate, which map observations into parameter estimates. 

* **Maximum likelihood estimation** of parameters, where we use **optimisation** to find parameter settings that make the the observations appear as likely as possible. We can see this as tweaking the parameters of some predefined model until they "best align" with the observations we have. This requires an iterative optimisation process, but it works for any model where the distribution has a known likelihood function \(that is we can compute how likely observations were to have been generated by that model\).

* **Bayesian, probabilistic** approaches explicitly encode belief about the behaviour of the mysterious entity using probability distributions. In Bayesian models, we assume a distribution over the parameters themselves, and consider the *parameters to be random variables*. We have an initial hypotheses for these parameters \("prior"\) and we use observations to update this belief to hone our estimate of the parameters to a tighter \(hopefully\) distribution \("posterior"\). Unlike the other methods, we do not estimate a single "parameter setting", but instead we always have a distribution over possible parameters which changes as data is observed. This is much more robust and arguably more coherent way to do inference, but it is harder to represent and harder to compute. We require both **priors** over parameters, and a **likelihood function** that will tell us how likely data is to have been generated under a particular parameter setting.

*Note: there are more general forms of Bayesian inference, like Approximate Bayesian Computation \(ABC\) which do not even require likelihood functions, just the ability to sample from distributions. We will not discuss these.*

### Estimators

Unlike discrete distributions, where the PMF can be estimated directly from observations using the empirical distribution \(as we did for Romeo and Juliet\), there is no analogous direct procedure for continuous distributions.

For many continuous distributions statisticians have developed **estimators**; functions that can be applied to sets of observations to estimate the most likely parameters of a probability density function defining a distribution that might have generated them.

The **form** of the distribution must be decided in advance \(for example, the assumption that the data has been generated by an approximately normal distribution\); this is usually called the **model**. The specific parameters can then be calculated under the assumption of this model.

### Direct estimation

One way of doing inference is to, if we assume a particular *form* of the distribution \(e.g. assume it is normal\), use **estimators** of **parameters** \(such as the mean and variance\) of this population distribution. These **estimators** are computed via **statistics** which are summarising functions we can apply to data. *These estimators need to specially derived for each specific kind of problem.*

For example, the arithmetic mean, and the standard deviation of a set of observed samples are **statistics** which are **estimators** of the parameters of μ and σ normal distribution. If we have observations \(believed to have been\) drawn from a normal distribution, we can estimate the parameters μ and σ of that distribution just by computing the mean and standard deviation.

### Standard estimators
#### Mean

The **arithmetic mean** is sum of sample values x<sub>1, x<sub>2</sub>, ..., x<sub>n</sub> divided by the number of values:  
μ^ = 1/N ∑ x<sub>i</sub> for i in range\(1,N\)

### Sample mean

The population mean is μ = expected X for a random variable X. It turns out the *arithmetic mean of the observed samples* or **sample mean**, which we write with a *little hat* μ^ is a good \(footnote: good is what statisticians would call "unbiased"\) estimator of the true population mean μ. As the number of samples increases, our estimate μ^ of the population mean μ gets better and better. 

It's important to separate the idea of  
* the population mean μ, which \(usually!\) exists but is not knowable directly. It is 
the expectation of the random variable E\[X\].
* the sample mean μ^ which is just the arithmetic average of samples we have seen \(e.g. computed via `np.mean(x, axis=0)`\)

The sample mean is a **statistic** \(a function of observations\) which is an **estimator** of the population mean \(which could be a **parameter** of a distribution\). Specific bounds can be put on this estimate; the standard error gives a measure of how close we expect that the arithmetic mean of samples is to the population mean, although the interpretation is not straightforward. 

The mean measures the **central tendency** of a collection of values. The **mean vector** generalises this to higher dimensions.

### Variance and standard deviation

The sample variance is the squared difference of each value of a sequence from the mean of  that sequence:

σ<sup>2</sup>^ = 1/N ∑ \(x<sub>i</sub>\-μ^\)<sup>2</sup> for i in range\(1,N\).

It is an estimator of the population variance, expected \(X\- expected X\)<sup>2</sup>

The sample standard deviation is just the square root of this value. 

σ^ = sqrt \[1/N ∑ \(x<sub>i</sub>\-μ^\)<sup>2</sup> for i in range\(1,N\)\]

The variance and the standard deviation measure the **spread** of a collection of values. The **covariance matrix** Σ generalises this idea to higher dimensions.

##### Relation to normal distribution

If we *assume* that our data is generated by a normal distribution, then the statistics **mean** μ^ and **variance** σ<sup>2</sup>^  estimate the parameters  μ, σ of that normal distribution, N\(μ, σ\). Even if the underlying process isn't exactly normal, it may well be close to being normal because of the Central Limit Theorem. And even if that doesn't apply, the mean and the variance are still useful *descriptive statistics*.

### Fitting

What does it mean to estimate the parameters of a normal distribution that might be creating app ratings? We are **fitting** a distribution, governed by a PDF, to a set of observations. In our discrete examples, we could fit a distribution simply by computing the empirical distribution \(assuming we had enough samples\). But estimating a PDF requires some structure, a space of functions with some parameterisation.

### Sampling from the model

We can draw samples from our fitted distribution, and compare them to our results. They won't be a very good representation, because the data we have is clearly not normal. But they show what our tame mysterious entity is producing, and let us assess our **modelling assumptions** -- that the app ratings were characterised by just a mean and standard deviation.

### Maximum likelihood estimation: estimation by optimisation

What if we don't have estimators, ready built to estimate the parameters that we want? How can we do inference? How can we fit distribution parameters to observations?

In many cases, we can compute the **likelihood** of a an observation being generated by a specific underlying random distribution. This is the **likelihood** that we saw earlier. For a PDF, the likelihood of a value x is just the value of the PDF at x: f<sub>X</sub>\(x\). The likelihood is a function of the data, under the assumption of some particular parameters. 

The likelihood of many *independent* observations is the product of the individual  likelihoods, and the log-likelihood is the sum of the individual log-likelihoods.

log L\(x<sub>1</sub>, ..., x<sub>n</sub>\) = ∑ log f<sub>X</sub>\(x<sub>i</sub>\)

Imagine we have a distribution which we *don't* know any **estimators** for the parameters. How could we estimate what they might be, given some data? We could write all of our parameters as vector θ; for example a normal distribution would have θ =\[μ, σ\].

### Optimisation solves all problems

Even though we don't have a fixed, closed form function to estimate the parameters, with a likelihood function we can apply optimisation to work out a parameter setting under which the data we *actually* observed was most likely. This corresponds to twiddling the knobs on our "mysterious entity" machine, until we find one that outputs the largest likelihood values when we feed in samples to it.

If the likelihood depends on some parameters of a distribution θ, then we write:

L\(θ \| x\)

Then, we could define an **objective function**; to maximise the log-likelihood, or equivalently to minimise the negative log\-likelihood.

θ<sup>\*</sup> = argmin L\(θ\)  
L\(θ\) = \-log L\(θ \| x<sub>1</sub>, ..., x<sub>n</sub>\) = \-∑ log f<sub>X</sub>\(x<sub>i</sub>;θ\),

assuming our f<sub>X</sub>\(x<sub>i</sub>\) can be written as f<sub>X</sub>\(x<sub>i</sub>;θ\) to represent the PDF of f with some specific choice of parameters given by θ.

### Maximum likelihood estimation

This is *very* similar to the approximation objective function we saw before \|f<sub>X</sub>\(x<sub>i</sub>;θ\)\|, but we have y=0 and we only have a scalar output from f so the norm is unnecessary. We already know how to solve this kind of problem; just optimise. This is called **maximum likelihood estimation** and is a general technique for determining parameters of a distribution which we don't know given some observations. It will find the **best** setting of parameters that would explain how the observations came to be.

If we're lucky, this will be differentiable and we can use gradient descent \(or stochastic gradient descent -- note that the objective function is a sum of simple sub\-objective functions\). If we're not, we can fall back to less efficient optimisers. We don't need special estimators in this case, as long as we can evaluate the PDF f<sub>X</sub>\(x<sub>i</sub>;θ\) for any setting of parameters θ. *This works for a much wider class of probability distributions*.

### Fitting a normal with MLE

We can for example look at the problem of estimating the mean and variance of a normal distribution from a set of \(assumed to be independent\) samples *without* using estimators; for example our app ratings. To do this, we need to be able to compute the likelihood for any given sample, and take the product \(or rather sum of log likelihoods\) for all of those samples. 

This gives us our objective function. If we flip the sign, so that we minimise the negative log\-likelihood, we will then search for the parameter vector that makes the data most likely. 

For a univariate normal distribution, the parameters are just μ and σ, so θ=\[μ, σ\].

In this case, of course, we *do* have estimators; but the procedure works just as well when we only have a likelihood function.

### A mixture model

But what if our model was more complicated that just a normal distribution? We could imagine that we model in some other way, perhaps that might be able to capture the fact that app B seems to have two "humps" on either side. One very simple model is a **mixture of Gaussians**, where we just say that we expect the PDF of the distribution we are trying to fit is a weighted combination \(convex sum\) of N different normal distributions \("components"\) N<sub>i</sub>\(μ<sub>i</sub>, σ<sub>i</sub>\), each with its *own* μ<sub>i</sub>, σ<sub>i</sub>, and with a weighting factor λ<sub>i</sub> that says how important this "component" is, where ∑ λ<sub>i</sub>=1. This lets us represent "humpy distributions.

This model lets us imagine that ratings might belong to one "cluster" or another. The placement and size of each cluster is given by the μ<sub>i</sub> and σ<sub>i</sub> for that component and λ<sub>i</sub> gives an idea of how likely data is to fall into that cluster.

We can easily plot the PDF of this function; it's just:  
f<sub>X</sub>\(x\) = ∑ λ<sub>i</sub> n<sub>X</sub>\(x; μ<sub>i</sub>, σ<sub>i</sub>\), where n<sub>X</sub>\(x; μ<sub>i</sub>, σ<sub>i</sub>\) = 1/Z e<sup>\(x\-μ<sub>i</sub>\)<sup>2</sup> / 2σ<sup>2</sup>}</sup> is the standard normal PDF function.

### Fitting mixtures

This is a much more plausible model of our app ratings, and might be a much better model. But how do we fit it? Even if we fix N in advance, we definitely don't have any direct estimators that can estimate the mean and standard deviation \(and weighting\) of a sum of normal PDFs. This simply isn't something we know how to do. 

**But** the \(log\) likelihood is trivial to write in code. For each observation x, we just compute the sum of the weighted PDFs for 
each component, and the result is likelihood for that observation. This is a function of the data L\(θ\|x\) , and our parameter vector is θ = \[μ<sub>1</sub>, σ<sub>1</sub>, λ<sub>1</sub>, μ<sub>2</sub>, σ<sub>2</sub>, λ<sub>2</sub>, ...\]$.

### Bayesian Inference

Bayesian inference involves thinking about the problem quite differently. Bayesians represent the *parameters* of the distribution they are estimating as random variables *themselves*, with distributions of their own. 

Prior distributions are defined over these parameters \(e.g. we might believe that the mean app rating could be any value 1.0-5.0 with equal probability\) and evidence arriving as updates is combined using Bayes' Rule to refine our belief about the distribution of the parameters. We again consider our distribution to be characterised by some parameter vector θ and we want to refine a distribution over possible θs.

We don't think about estimators, or their sampling distributions, and it doesn't make sense to talk about finding the best parameter setting; we can only have *beliefs* in parameter settings which must be represented probabilistically. We do not seek to find the most likely parameter setting \(as in direct estimation or MLE\), but to infer a distribution over possible parameter settings *compatible with the data*. 

We talk about inferring a **posterior** distribution over the parameters, given some **prior** belief and some **evidence**. We assume that we have a **likelihood function** P\(D\|θ\), and a prior over parameters P\(θ\) and we can then use Bayes Rule in the form:

P\(θ\|D\) = P\(D\|θ\) P\(θ\) / P\(D\)

which gives us a new distribution over θ given some observations. Bayes' rule applies just as well to continuous distributions as to discrete ones, but computations in "closed form" \(i.e. algebraically\) are much harder. 

This *can* be done in closed form to find P\(θ\|D\) in certain cases, but the algebra is often complex and the model choices are limited; we will not discuss how to do this. When it is possible, it is, however, much more computationally efficient. Instead we will approach this from a computational perspective and find a way to draw *samples* from the posterior distribution P\(θ\|D\).

### Inference

How can we compute the posterior distribution P\(θ\|D\)? We won't discuss how to find this in closed form \(as a function\) -- this is sometimes possible, but mathematically involved because we need to deal with distributions over $θ$ -- but rather how to draw samples from this posterior, given a prior and a likelihood and some observations. 

There is a huge literature on how to solve this problem, which has a few nasty parts:
* P\(D\|θ\) needs to be computed for a **distribution** over θ, not just some numbers. It's no good to just compute the probability for one specific θ; we have to work with distribution functions.
* P\(D\) = ∫<sub>θ</sub> P\(D\|θ\)P\(θ\) which is likely intractable.

#### Making it tractable

There are lots of ways this can be simplified to make it possible to solve. We are going to use two:

##### Samples will do

We often can't compute P\(θ\|D\) because we don't know how to do operations on products of functions. But it's often trivial for *specific, concrete* values of θ. For example, for a given fixed θ we can compute both the likelihood and the prior of that specific example.

This leads us to the idea of **drawing samples** from the posterior distribution P\(θ\|D\), instead of trying to compute the distribution exactly. 

##### Relative probability only

We can make a simplifying assumption: we only care about the *relative* probability of different parameter settings with the data that we actually have, D. That is we have

P\(θ\|D\) ∝ P\(D\|θ\)P\(θ\) and ignore the fact that this is the posterior scaled by some unknown constant  Z=1/P\(D\). This only makes sense because we are only considering one model with one set of data in this example.

### Markov Chain Monte Carlo

We can implement a procedure to sample from the \(relative\) posterior distribution via a very simple modification of the *simulated annealing* algorithm. 

This defines a random walk through the space of parameter settings, proposing small random tweaks to the parameter settings, and accepting "jumps" if they make the estimate more likely, or with a probability proportional to the change in P\(D\|θ\)P\(θ\) if not. The advantage of this approach is that we can work with *definite samples* from θ and we don't have to do any tricky integrals. This approach is called **Markov Chain Monte Carlo**

All we require is a way of evaluating P\(θ\) \(prior\) and P\(D\|θ\) \(likelihood\) for any specific θ.

### MCMC in practice: sampling issues

We will use Markov Chain Monte Carlo to solve the Bayesian inference problem. The **great thing** about MCMC approaches is that you can basically write down your model and then run inference directly. There is no need to derive complex approximations, or to restrict ourselves to limited models for which we can compute answers analytically. Essentially, no maths by hand; everything is done algorithmically.

MCMC allows us to draw samples from any distribution P\(X=x\) *that we can't sample from directly*. In particular, we will be able to sample from the posterior distribution over parameters.  

The **bad thing** about MCMC approaches is that, even though it will do the "right thing" *asymptotically*, the choice of sampling strategy has a very large influence for the kind of sample runs that are practical to execute. Bayesian inference should depend only on the priors and the evidence observed; but MCMC approaches also depend on the sampling strategy used to approximate the posterior. 

### What distribution are we sampling from?

In the case of Bayesian inference P\(θ\|D\) = P\(D\|θ\) P\(θ\) / P\(D\) = P\(D\|θ\)P\(θ\) / ∫<sub>θ</sub> P\(D\|θ\)P\(θ\).
* P\(θ\|D\) is the posterior, the distribution over the parameters θ given the data \(observations\) D, using:
* the likelihood P\(D\|θ\), 
* prior P\(θ\) and 
* evidence P\(D\). 

In other words, what is the distribution over the parameters given the observations and the prior? If we assume, as above, that we don't care about P\(D\), because we are only comparing different possible values of θ, then we can draw samples from a distribution proportional to P\(D\|θ\)P\(θ\).

### Metropolis-Hastings

Metropolis\-Hastings \(or just plain Metropolis\) is a wonderfully elegant and relatively effective way of doing this MCMC algorithm, and is able to work in high\-dimensional spaces \(i.e. when θ is complicated\). 

Metropolis sampling uses a simple auxiliary distribution called the **proposal distribution** Q\(θ'\|θ\) to help draw samples from an intractable posterior distribution P\(θ\|D\). This is analogous to what we called the **neighbourhood function** in the optimisation section.

Metropolis-Hastings uses this to **wander around** in the distribution space, accepting jumps to new positions using Q\(θ'\|θ\) to randomly sample the space of P\(θ\|D\).  This random walk \(a **Markov chain**, because we make a random jump conditioned only on where we currently are\) is a the "Markov Chain" bit of "Markov Chain Monte Carlo".

This is just like the simulated annealing algorithm, except now there is a function f<sub>X</sub>\(θ\) which makes some steps more likely than others instead of a likelihood function. We just take our current position θ, and propose a new position θ', that is a random sample drawn from Q\(θ'\|θ\). Often this is something very simple like a normal distribution with mean x and some preset $σ$: Q\(θ'\|θ\) = N\(θ, σ'\)

### Trace

The history of accepted samples of an MCMC process is called the **trace**. We can estimate model parameters by looking at the histogram of the **trace**, for example. The trace is the sequence of samples \[x<sup>\(1\)</sup>, x<sup>\(2\)</sup>, x<sup>\(3\)</sup>, ... x<sup>\(n\)</sup>\]$, \(approximately\) drawn from the **posterior** distribution P\(θ\|D\) via MCMC.

### Predictive posterior: sampling from the model

The **predictive posterior** is the *distribution over observations* we would expect to see; predictions of future samples. This means drawing samples from the model, while integrating over parameters from the posterior. By sampling from the predictive posterior, we are generating new synthetic data that should have the same statistical properties as the data \(if our model is good\).

We can do this with a two step, nested process:

* for n repetitions
* draw samples from our posterior distribution over parameters to give us a concrete distribution
    * for m repetitions
        * draw samples from this concrete distribution

### Linear regression

**Linear regression** is the fitting of a line to observed data. It assumes the mysterious entity generates data where one of the observed variables is scaled and shifted version of another observed variable, corrupted by some noise; a linear relationship. It is a very helpful lens through which different approaches to data modelling can be seen; it is pretty much the simplest useful model of data with relationships, and the techniques we use easily generalise from linear models to more powerful representations.

The problems is to estimate what that scaling and shifting is. In a simple 2D case, this is the gradient m and offset c in the equation y=mx\+c. It can be directly generalised to higher dimensions to find A and b in y = Ax \+ b, but we'll use the simple "high school" y=mx\+c case for simplicity.

We assume that we will fit a line to *noisy* data. That is the process that we assume that is generating the data is y=mx\+c\+ϵ
, where ϵ is some noise term. We have to make assumptions about the distribution of ϵ in order to make inferences about the parameters.

One simple assumption is that ϵ ∼ N\(0, σ<sup>2</sup>\), i.e. that we have normally distributed variations in our measurements. So our full equation is:

y=mx\+c\+N\(0, σ<sup>2</sup>\),

or equivalently, putting the mx\+c as the mean of the normal distribution:

y ∼ N\(mx\+c, σ<sup>2</sup>)

Note that we assume that y is a random variable, x is known, and that m, c, σ are parameters that we wish to infer from a collection of observations.

Our problem is: given just the inputs x and return values$y, what are the values of the *other* argument θ.

### Linear regression via direct optimisation

We saw how this problem could be solved as a **function approximation** problem using optimisation. We can write an objective function:

L\(θ\) = \|f\(x;θ\)\-y\|, where θ=\[m,c\]$ and f\(x;θ\) = θ<sub>0</sub> x \+ θ<sub>1</sub>. 

If we choose the squared Euclidean norm, then we have, for the simple y=mx\+c case:

L\(θ\) = \|f(x;θ)\-y\|$$
L\(θ\) = \|θ<sub>0</sub> x + θ<sub>1</sub> \- y\|<sup>2</sup><sub>2</sub> = \(θ<sub>0</sub> x + θ<sub>1</sub> \- y\)<sup>2</sup>, 

which we can easily minimise, e.g. by gradient descent, since computing ∇ L\(θ\) turns out to be easy.  This is **ordinary linear least\-squares**.

*Linear least squares tried to make the size of the squares nestled between the line and data points as small as possible*

In fact, we can find a closed form solution to this problem, without doing any iterative optimisation. This is because we have an **estimator** that gives us an estimate of the parameters of the line fit directly from observations. We can derive this, for example, by setting $∇ L(θ)=0$ and solving directly (high-school optimisation).

### Linear regression via maximum likelihood estimation
We could also consider this to be a problem of inference. We could explicitly assume that we are observing samples from a distribution whose parameters we wish to estimate. This is a **maximum likelihood approach**. This requires that we can write down the problem in terms of the distribution of random variables.

If we assume that "errors" are normally distributed values which are corrupting a perfect $y=mx+c$ relationship, we might have a model Y ∼ N\(mx+c, σ^2\); Y has mean mx\+c and some standard deviation σ. 

We can write this as a maximum likelihood problem (MLE), where we maximise L\(θ\|x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>,y<sub>2</sub>, ..., x<sub>n</sub>, y<sub>n</sub>\). To avoid underflow, we work with the log of the likelihood and minimise the negative log\-likelihood. The log\-likelihood of  independent samples x<sub>i</sub>  is given by:

log L\(θ\|x<sub>1</sub>, y<sub>1</sub>,  ..., x<sub>n</sub>, y<sub>n</sub>\) = log ∏ f<sub>Y</sub>\(x<sub>i</sub>, y<sub>i</sub>) = ∑ log f<sub>Y</sub>(x<sub>i</sub>, y<sub>i</sub>\), 

f<sub>Y</sub>(x<sub>i, y<sub>i) = \frac{1}{Z}\, e<sup>\-\(y<sub>i</sub> \- μ\)<sup>2</sup> / 2σ<sup>2</sup></sub>, μ = mx<sub>i\+c

We can then minimise the negative log-likelihood to find the "most likely" setting for θ=\[m,c,σ\], which \(if we feel like writing out long equations in LaTeX\), we could write as an objective function:

L\(θ\) = \-∑ log 1 / Z, e<sup>\-(y<sub>i - θ<sub>0</sub> x<sub>i + θ<sub>1</sub>\) / <sup>2</sup> / 2 θ<sub>2</sub><sup>2</sup></sub>,

In the case where we have normally distributed noise for linear regression, this turns out to be *exactly* equivalent to the direct optimisation with linear least-squares, although we will also find the standard deviation of the error σ in addition to m and c. This is **maximum likelihood linear regression**.

*Maximum likelihood estimation tried to find parameters of a line that made the observations likely*

### Bayesian linear regression 

What if we wanted to know how sure our estimates of m and c \(and σ\) were? MLE will tell us the *most likely setting*, but it won't tell us the possible settings that are compatible with the data.

The Bayesian approach is to let the parameters themselves by random variables. We don't want to optimise. We don't want to find the most likely parameters. We instead want to derive a belief about the parameters as a probability distribution. This is what Bayesians do; they represent belief with probability.

So we can write θ = \[m,c,σ\] as a random variable, and try and infer the distribution over it. We can do this using Bayes' rule.  Writing in the form \(D=data, H=hypothesis; hypothesised parameter settings\):

P\(H\|D\) = P\(D\|H\) P\(H\) / P\(D\)

Assuming our hypotheses H are parameterised by θ, then we want to know P\(θ\|D\) = P\(D\|θ\)P\(θ\) / P\(D\), where D stands for the data \[\(x<sub>1</sub>, y<sub>1</sub>\), \(x<sub>2</sub>, y<sub>2</sub>\), ..., \(x<sub>n</sub>, y<sub>n</sub>\)\]. In linear regression θ can be seen as the hypothesis that the data was generated by a line with parameters specified by θ.

We need:  
* a **prior** over the parameters P\(θ\). An initial belief about the possible gradient m, offset c and noise level σ, in the linear regression case.
* a way of calculating the **likelihood** P\(D\|θ\). 
* a way of combining these using Bayes Rule. In general this is impossible to compute exactly \(in particular the P\(D\) term is often intractable\), but we could sample from it using **Markov Chain Monte Carlo**, for example.

This will give us samples from the posterior distribution of P\(θ\|D\), so we can see how sure we should be about our beliefs about the parameters of the mysterious entity.

*Bayesian regression tries to update a distribution over line parameters given evidence*




