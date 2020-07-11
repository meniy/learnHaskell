-- function
doubleMe x = x + x
-- function con due parametri
doubleUs x y = x*2 + y*2
-- lista
lostNumbers = [4,6,15,16,23,42]

-- condizione
doubleSmallNumber x = if x > 100 -- deve sempre tornare qualcosa
  then x
  else x*2

-- head
head' = head [5,4,3,2,1] -- 1
-- tail
tail' = tail [5,4,3,2,1] -- [4,3,2,1]
-- last
last' = last [5,4,3,2,1] -- 1
-- init
init' = init [5,4,3,2,1] -- [5,4,3,2,1] => [5,4,3,2]
-- errore head
erroreHead' = head [] -- *** Exception: Prelude.head: empty list
--length
length' = length [5,4,3,2,1]
-- null
nullFalse = null [1,2,3] --False
nullTrue = null [] --True
-- reverse
reverse' = reverse [5,4,3] -- [3,4,5]
-- take
take' = take 2 [5,4,3,2,1] -- [5,4]
takeZero = take 0 [5,4,3,2,1] -- []
-- drop
drop' = drop 3 [8,4,2,1,5,6] -- [1,5,6]
-- maximum
maximum' = maximum [1,6,3] -- 6
-- minimum
minimum' = minimum [1,6,3] -- 1
-- product
product' = product [6,2,1,2] --24
-- sum
sum' = sum [5,2,1,6,3,2,5,7] --31
-- elem
elem' = 4 `elem` [3,4,5,6] -- True
elemFalse = 10 `elem` [3,4,5,6] -- False
-- mod
mod' = 4 `mod` 2
-- range
range' = [1..20] -- [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
alphabetRange = ['a'..'z'] --"abcdefghijklmnopqrstuvwxyz"
upperCaseAlphabetRange = ['A'..'A'] --"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
subRange = ['K'..'Z'] --"KLMNOPQRSTUVWXYZ"
stepRange = [2,4..20] -- [2,4,6,8,10,12,14,16,18,20]

-- cycle
cycle' = take 10 (cycle [1,2,3]) -- [1,2,3,1,2,3,1,2,3,1]
-- repeat
repeat' = take 10 (repeat 5) -- [5,5,5,5,5,5,5,5,5,5]
-- replicate
replicate' = replicate 3 10 -- [10,10,10]

-- list comprehension
-- Esempio in matematica: S={2*x | x in N, x <= 10}
comprehension = [x*2 | x<-[1..10]] -- [2,4,6,8,10,12,14,16,18,20]
conditionalComprehension = [x*2 | x<-[1..10],x*2>=12] -- [12,14,16,18,20]

boomBangs xs = [ if x < 10 then "BOOM!" else "BANG!" | x <- xs, odd x] --boomBangs [7..13] => ["BOOM!","BOOM!","BANG!","BANG!"]
allNumberNotSomeOne=[ x | x <- [10..20], x /= 13, x /= 15, x /= 19]   -- [10,11,12,14,16,17,18,20]

-- double loop
comprehensionDoubleLoop =[ x*y | x <- [2,5,10], y <- [8,10,11]] -- [16,20,22,40,50,55,80,100,110]
comprehensionDoubleLoopCond =[ x*y | x <- [2,5,10], y <- [8,10,11], x*y > 50] -- [55,80,100,110]

-- _
len xs = sum [1| _ <- xs] -- xs=[1,2,1,5] => 4

-- remove uppercase
removeNonUppercase st = [c | c <-st, c `elem` ['A'..'Z']]

-- TUPLE

-- fst
fst'=fst(8,11) -- 8
--snd
snd'=snd(8,11) --11

-- fstError=fst(8,11) --Error
--zip
zip'=zip [1,2,3] [2,4,6] -- [(1,2),(2,4),(3,6)]
-- zip with different lengths of the lists
zipDiffLeng=zip [1,2,3] ['a','b'] --[(1,a),(2,b)]

-- all possible triangles  with sides under or equal to 10
-- side b isn't larger than the hypothenuse and that side a isn't larger than side b.
-- the perimeter is 24
triangles = [ (a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2, a+b+c == 24 ]

-- type declaration
removeNonUppercase2 :: [Char] -> [Char]
removeNonUppercase2 st = [c | c <-st, c `elem` ['A'..'Z']]

addThree :: Int -> Int -> Int -> Int
addThree x y z = x + y + z

--read
--Haskell is a statically typed language, it has to know all the types before the code is compiled
read'=read "5"+1
read2'=read "[1,2]"++[3]
--read "5" Error!
read3=read "5"::Int
read4=read "(3,'a')"::(Int,Char)

-- Syntax in functions
factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n-1)

-- Pattern matching with tuples
addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

first2' :: (a, b, c) -> a
first2' (x, _, _) = x
-- pattern match in list comprehensions
xs = [(1,3), (4,3), (2,4), (5,3), (5,6), (3,1)]
pattermatchListCon=[a+b | (a,b) <- xs]--[4,7,6,8,11,4]

myHead' :: [a]-> a
myHead' [] = error "Can't call head on an empty list, dummy!"
myHead' (x:_) = x

tell :: (Show a) => [a] -> String
tell [] = "The list is empty"
tell (x:[]) = "The list has one element: "++ show x
tell (x:y:[])= "The list has two elements: "++ show x++ " and " ++ show y
tell (x:y:_) = "This list is long. First: "++show x++" Second: "++show y
-- pattern matching and a little recursion
length2' :: (Num b) => [a] -> b
length2' [] = 0
length2' (_:xs) = 1 + length2' xs --[2000,3,4,5,2] => 5
-- sum
mySum :: (Num a) => [a] -> a
mySum [] = 0
mySum (xs:xxs) = xs + mySum xxs
-- as patterns
capital :: String -> String
capital "" = "Empty string, whoops!"
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]

-- bmiTell :: (RealFloat a) => a -> String
-- bmiTell bmi
--     | bmi <= 18.5 = "You're underweight, you emo, you!"
--     | bmi <= 25.0 = "You're supposedly normal. Pffft, I bet you're ugly!"
--     | bmi <= 30.0 = "You're fat! Lose some weight, fatty!"
--     | otherwise   = "You're a whale, congratulations!"


-- bmiTell :: (RealFloat a) => a -> a -> String
-- bmiTell weight height
--     | weight / height ^ 2 <= 18.5 = "You're underweight, you emo, you!"
--     | weight / height ^ 2 <= 25.0 = "You're supposedly normal. Pffft, I bet you're ugly!"
--     | weight / height ^ 2 <= 30.0 = "You're fat! Lose some weight, fatty!"
--     | otherwise                 = "You're a whale, congratulations!"
--     bmiTell :: (RealFloat a) => a -> a -> String

-- no repeat
-- bmiTell weight height
--     | bmi <= 18.5 = "You're underweight, you emo, you!"
--     | bmi <= 25.0 = "You're supposedly normal. Pffft, I bet you're ugly!"
--     | bmi <= 30.0 = "You're fat! Lose some weight, fatty!"
--     | otherwise   = "You're a whale, congratulations!"
--     where bmi = weight / height ^ 2


bmiTell :: (RealFloat a) => a -> a -> String
bmiTell weight height
    | bmi <= skinny = "You're underweight, you emo, you!"
    | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"
    | bmi <= fat    = "You're fat! Lose some weight, fatty!"
    | otherwise     = "You're a whale, congratulations!"
    where bmi = weight / height ^ 2
          skinny = 18.5
          normal = 25.0
          fat = 30.0

myMax :: (Ord a) => a -> a -> a
myMax a b
    | a > b = a
    | otherwise = b

initials :: String -> String -> String
initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
    where (f:_) = firstname
          (l:_) = lastname

calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmi w h | (w, h) <- xs]
    where bmi weight height = weight / height ^ 2

-- let bindings can be used for pattern matching
cylinder :: (RealFloat a) => a -> a -> a
cylinder r h =
    let sideArea = 2 * pi * r * h
        topArea = pi * r ^2
    in  sideArea + 2 * topArea

-- let zoot x y z = x * y + z -- compilatore da errore, da prompt no

-- case expressions
-- case expression of pattern -> result
--                    pattern -> result
--                    pattern -> result
--                    ...
myhead' :: [a] -> a
myhead' xs = case xs of [] -> error "No head for empty lists!"
                        (x:_) -> x
-- recurion
-- finbonacci
-- edge condition F(0)=0 F(1)=1

-- maximum
myMaximum :: (Ord a) => [a]-> a
myMaximum [] = error "list is empty"
myMaximum [x]=x -- edge condition
myMaximum (x:xs)
  | x>maxTail = x
  | otherwise = maxTail
  where maxTail = myMaximum xs


maximum3 :: (Ord a) => [a] -> a
maximum3 [] = error "maximum of empty list"
maximum3 [x] = x
maximum3 (x:xs) = max x (maximum3 xs)

--replicate with recursion
recuReplicate :: (Num i, Ord i) => i -> a -> [a]
recuReplicate n x
  | n<=0 = []
  | otherwise=x:recuReplicate (n-1) x

-- take with recursion
recuTake ::(Num i, Ord i) => i-> [a] -> [a]
recuTake n x  -- 2 [5,4,3,2] => [5,4]
  | n <= 0 = []
recuTake _ [] = [] -- indipendetemente da n se il vettore vuoto ritorna vuoto
recuTake n (x:xs) = x : recuTake (n-1) xs

-- reverse with recursion
recReverse' :: [a] -> [a]
recReverse' [] = []
recReverse' (x:xs) = recReverse' xs ++ [x]

-- zip
recZip' :: [a] -> [b] -> [(a,b)]
recZip' _ [] = []
recZip' [] _ = []
recZip' (x:xs) (y:ys) = (x,y):recZip' xs ys


-- HOF
-- Curried Functions
--  All the functions that accepted several parameters so far have been curried functions
-- max :: (Ord a) => a -> a -> a.
-- That can also be written as
-- max :: (Ord a) => a -> (a -> a)
-- That could be read as:
-- max takes an a and returns (that's the ->) a function that takes an a and returns an a.
-- That's why the return type and the parameters of functions are all simply separated with arrows.


-- multThree :: (Num a) => a -> a -> a -> a
-- multThree x y z = x * y * z

-- What really happens when we do multThree 3 5 9 or ((multThree 3) 5) 9?
-- First, 3 is applied to multThree, because they're separated by a space.
-- That creates a function that takes one parameter and returns a function.
-- So then 5 is applied to that, which creates a function that will take a parameter and multiply it by 15.
-- 9 is applied to that function and the result is 135 or something.
-- Remember that this function's type could also be written as
-- multThree :: (Num a) => a -> (a -> (a -> a)).
-- The thing before the -> is the parameter that a function takes and the thing after it is what it returns.
-- So our function takes an a and returns a function of type
-- (Num a) => a -> (a -> a).
-- Similarly, this function takes an a and returns a function of type
-- (Num a) => a -> a.
-- And this function, finally, just takes an a and returns an a.


-- compareWithHundred :: (Num a, Ord a) => a -> Ordering
-- compareWithHundred x = compare 100 x
-- equivalent to
compareWithHundred :: (Num a, Ord a) => a -> Ordering
compareWithHundred = compare 100

-- surround with parentheses
-- That creates a function that takes one parameter and
-- then applies it to the side that's missing an operand
divideByTen :: (Floating a) => a -> a
divideByTen = (/10)   -- divideByTen 200 =>20 or divideByTen 50=>5

isUpperAlphanum :: Char -> Bool
isUpperAlphanum = (`elem` ['A'..'Z'])

-- Warning
-- divideByTen :: (Floating a) => a -> a
-- divideByTen = (-4) -- minus four!!

subByFour :: (Floating a) => a -> a
subByFour = (subtract 4)

applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

-- map
--takes a function and a list and applies that function to every element in the list, producing a new list.
map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

-- -- filter
-- filter' :: (a-> Bool) -> [a] -> [a]
-- filter' _,[]=[]
-- filter' p (x:xs)
--   | p x       = x : filter' p xs -- if p x is True elements gets included in the new list, otherwise not
--   | otherwise = filter p xs

-- quicksort with filters
quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
    let smallerSorted = quicksort (filter (<=x) xs)
        biggerSorted = quicksort (filter (>x) xs)
    in  smallerSorted ++ [x] ++ biggerSorted

--takeWhile
-- It takes a predicate and a list and then goes from the beginning of the list and returns its elements while the predicate holds true
--Once an element is found for which the predicate doesn't hold, it stops.
-- we wanted to get the first word of the string
-- Next up, we're going to find the sum of all odd squares that are smaller than 10,000. But first, because we'll be using it in our solution, we're going to introduce the takeWhile function. It takes a predicate and a list and then goes from the beginning of the list and returns its elements while the predicate holds true. Once an element is found for which the predicate doesn't hold, it stops.
-- If we wanted to get the first word of the string "elephants know how to party", we could do
-- takeWhile (/=' ')
-- sum (takeWhile (<10000) (filter odd (map (^2) [1..])))


--For our next problem, we'll be dealing with Collatz sequences.
-- We take a natural number. If that number is even, we divide it by two. If it's odd, we multiply it by 3 and then add 1 to that
-- We take the resulting number and apply the same thing to it, which produces a new number and so on
-- 13, we get this sequence: 13, 40, 20, 10, 5, 16, 8, 4, 2, 1
-- 13*3+1=40, 40/2=20,20/2=10,10/2=5,5*3+1=16,16/2=8,8/2=4,4/2=2,2/2=1

chain :: (Integral a) => a -> [a]
chain 1 = [1]
chain n
    | even n =  n:chain (n `div` 2)
    | odd n  =  n:chain (n*3 + 1)

-- Now what we want to know is this:
-- for all starting numbers between 1 and 100, how many chains have a length greater than 15?
-- sum (takeWhile (<10000) (filter odd (map (^2) [1..])))
numLongChains :: Int
numLongChains = length (filter isLong ( map chain [1..100]))
  where isLong xs = length xs >15
