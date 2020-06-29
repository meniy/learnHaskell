# learn-haskell

platform: (https://www.haskell.org/platform/#linux-fedora)[haskell-platform-fedora]
book: (http://learnyouahaskell.com/)[Learn You a Haskell]

Done:
- [x] Introduction
  - [x] About this tutorial
  - [x] So what's Haskell?
  - [x] What you need to dive in  

- [x] Starting Out
  - [x] Ready, set, go!
  - [x] Baby's first functions
  - [x] An intro to lists
  - [x] Texas ranges
  - [x] I'm a list comprehension
  - [x] Tuples

- [x] Types and Typeclasses
  - [x] Believe the type
  - [x] Type variables
  - [x] Typeclasses 101

- [x] Syntax in Functions
  - [x] Pattern matching
  - [x] Guards, guards!
  - [x] Where!?
  - [x] Let it be
  - [x] Case expressions

- [x] Recursion
  - [x] Hello recursion!
  - [x] Maximum awesome
  - [x] A few more recursive functions
  - [x] Quick, sort!
  - [x] Thinking recursively


Todo:
- [] Higher Order Functions
  - [] Curried functions
  - [] Some higher-orderism is in order
  - [] Maps and filters
  - [] Lambdas
  - [] Only folds and horses
  - [] Function application with $
  - [] Function composition
- [] Modules
  - [] Loading modules
  - [] Data.List
  - [] Data.Char
  - [] Data.Map
  - [] Data.Set
  - [] Making our own modules
  - [] Making Our Own Types and Typeclasses
  - [] Algebraic data types intro
  - [] Record syntax
  - [] Type parameters
  - [] Derived instances
  - [] Type synonyms
  - [] Recursive data structures
  - [] Typeclasses 102
  - [] A yes-no typeclass
  - [] The Functor typeclass
  - [] Kinds and some type-foo
- [] Input and Output
  - [] Hello, world!
  - [] Files and streams
  - [] Command line arguments
  - [] Randomness
  - [] Bytestrings
  - [] Exceptions
- [] Functionally Solving Problems
  - [] Reverse Polish notation calculator
  - [] Heathrow to London
- [] Functors, Applicative Functors and Monoids
  - [] Functors redux
  - [] Applicative functors
  - [] The newtype keyword
  - [] Monoids
- [] A Fistful of Monads
  - [] Getting our feet wet with Maybe
  - [] The Monad type class
  - [] Walk the line
  - [] do notation
  - [] The list monad
  - [] Monad laws
- [] For a Few Monads More
  - [] Writer? I hardly know her!
  - [] Reader? Ugh, not this joke again.
  - [] Tasteful stateful computations
  - [] Error error on the wall
  - [] Some useful monadic functions
  - [] Making monads
- [] Zippers
  - [] Taking a walk
  - [] A trail of breadcrumbs
  - [] Focusing on lists
  - [] A very simple file system
  - [] Watch your step

### Avvio interactive mode

```bash
$ ghci
```

```bash
$ :set prompt "ghci> "
```

# Creazione script, compiling e run
Il file va salvato in formato `.hs`
e in `GHCI` eseguire: `:l [FILE_NAME]`, da questo momento è possibile invocare le funzioni contenute nel file

**NB**: `:r` è equivalente perchè ricarica lo script corrente
