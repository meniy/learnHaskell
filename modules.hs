-- All the functions, types and typeclasses that we've dealt with so far were part of the Prelude module
-- import <module name>
-- This must be done before defining any functions, so imports are usually done at the top of the file.
import Data.List
--how many unique elements a list has.
numUniques :: (Eq a) => [a] -> Int
numUniques = length . nub
-- nub is a function defined in Data.List that takes a list and weeds out duplicate elements.
--  put the functions of modules into the global namespace when using GHCI
-- ghci> :m + Data.List
--  several modules inside GHCI
-- ghci> :m + Data.List Data.Map Data.Set
--  If we wanted to import only the nub and sort functions from Data.List, we'd do this:
-- import Data.List (nub, sort)
-- Say we already have our own function that's called nub and we want to import all the functions from Data.List except the nub function:
--import Data.List hiding (nub)

-- Another way of dealing with name clashes
-- is to do qualified imports.
-- The Data.Map module, which offers a data structure
--  for looking up values by key, exports a bunch of functions with the same name as Prelude functions, like filter or null. So when we import Data.Map and then call filter, Haskell won't know which function to use. Here's how we solve this:

import qualified Data.Map

-- This makes it so that if we want to reference Data.Map's filter function, we have to do Data.Map.filter, whereas just filter still refers to the normal filter we all know and love. But typing out Data.Map in front of every function from that module is kind of tedious. That's why we can rename the qualified import to something shorter:

import qualified Data.Map as M
-- Now, to reference Data.Map's filter function, we just use M.filter.

-- To search for functions or to find out where they're located, use Hoogle. It's a really awesome Haskell search engine, you can search by name, module name or even type signature.

-- Data.list
-- Map
-- filter
-- intersperse
intersperse '.' "MONKEY"
-- "M.O.N.K.E.Y"
intersperse 0 [1,2,3,4,5,6]
-- [1,0,2,0,3,0,4,0,5,0,6]

-- intercalate
intercalate " " ["hey","there","guys"]
-- "hey there guys"
intercalate [0,0,0] [[1,2,3],[4,5,6],[7,8,9]]
-- [1,2,3,0,0,0,4,5,6,0,0,0,7,8,9]
-- traspose
transpose [[1,2,3],[4,5,6],[7,8,9]]
--[[1,4,7],[2,5,8],[3,6,9]]
transpose ["hey","there","guys"]
--["htg","ehu","yey","rs","e"]
--foldl' and foldl1', are stricter versions of their respective lazy incarnations So if you ever get stack overflow errors when doing lazy folds, try switching to their strict versions.
-- concat
concat ["foo","bar","car"]
--"foobarcar"
concat [[3,4,5],[2,3,4],[2,1,1]]
--[3,4,5,2,3,4,2,1,1]

concatMap (replicate 4) [1..3]
--[1,1,1,1,2,2,2,2,3,3,3,3]

and $ map (>4) [5,6,7,8]  -- take a list of boolean values and returns True only if all values are True
-- True
and $ map (==4) [4,4,4,3,4]
-- False

or $ map (==4) [2,3,4,5,6,1]
--True
or $ map (>4) [1,2,3]
--False

any (==4) [2,3,5,6,1,4]
--True
all (>4) [6,9,10]
--True

all (`elem` ['A'..'Z']) "HEYGUYSwhatsup"
--False
any (`elem` ['A'..'Z']) "HEYGUYSwhatsup"
--True
