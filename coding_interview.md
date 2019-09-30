Table of Content
- [FAQ](#faq)
- [Walking through a Problem](#walking-through-a-problem)
- [Optimize & Solve Technique](#optimize--solve-technique)
- [Codeforces](#codeforces)
- [Interviewbit](#interviewbit)
- [Cracking the coding interview](#cracking-the-coding-interview)
- [Leetcode](#leetcode)
  - [Classification](#classification)
  - [Math](#math)
  - [Sorting](#sorting)
  - [String](#string)
  - [Array](#array)
  - [Matrix](#matrix)
  - [Binary Search](#binary-search)
  - [Linked List](#linked-list)
  - [Stack and Queue](#stack-and-queue)
  - [Heap (Priority Queue)](#heap-priority-queue)
  - [Cache](#cache)
  - [Tree](#tree)
  - [Trie (Prefix Tree)](#trie-prefix-tree)
  - [BFS & DFS](#bfs--dfs)
  - [Dynamic Programming](#dynamic-programming)
  - [Backtracking](#backtracking)
  - [Graph](#graph)
  - [Bit Manipulation](#bit-manipulation)
  - [Union Field](#union-field)
  - [Retry List](#retry-list)

## FAQ
  * [What is tail recursion?](https://stackoverflow.com/questions/33923/what-is-tail-recursion)

## [Walking through a Problem](http://www.crackingthecodinginterview.com/resources.html)
  1. Listen Carefully
     * For example:
       * Given two arrays that are **sorted** ...
  2. Draw an Example
     * **There'a an art to drawing an example though**.
     * Most examples are too small or are special cases.
  3. State a Brute Force
     * Even if it's obvious for you, it's not ncecessarily obvious for all candidates. You don't want your interviewer to think you're struggling to see even the wasy solution.
  4. Optimize
     * Look for any **unused information**.
     * Use a fresh example
     * Solve it "incorrectly"
     * Make **time vs. space** tradeoff
     * **Precompute** information
       * Is there a way that you can reorganize the data (sortig, etc.)
     * Use a **hash table**
     * Thank about the best conceivable runtime
  5. Walk Through
     * Whiteboard coding is **slow**, you need to make sure that you get is as close to "perfect" in the beginning as possible.
  6. Implement
     * Modularized code
     * Error checks
       * A good compromise here is to add a todo and then just explain out loud what you'd like to test
     * Use other classes/structs where appropriate
     * Good variable names
  7. Test
     * Conceptual test
       * Does the code do what you think it should do?
     * Weird looking code
       * Doulbe check that line of code that says x = length -2.
     * Hot spots
       * Like base cases in recursive code. Integer division. Null nodes in binary tree.
     * Small test cases
     * Special cases
       * null of single element values.

## [Optimize & Solve Technique](http://www.crackingthecodinginterview.com/resources.html)
1. Look for **BUD**
   * **B**ottlenecks *
    * For example, suppose you have a two-step algorithm where you first sort the array and then you find elements with a particular property.
      * The first step is O(nlong(n)) and the second step if O(n).
      * Perhaps you could reduce the second step to O(1), but would it matter? Not too much as o(nlong(n)) is the bottleneck
   * **U**nnecessary work
   * **D**uplicated work
2. Do it yourself
3. **Simplify and Generalize** *
   * First, we simplify or tweak some constraint, such as the data type. Then we solve this new simplified version of the problem. Finally, once we have an algorithm for the simplified problem, we try to adapt it for the more complex version.
4. **Base case and Build** **
   * Solve the problem first for a base case (e.g., n=1) and then try to build up from there. **When we get to more complex cases (often n=3 or n=4), we try to build those using the prior solution**.
5. Data Structure Brainstorm **
   * Try to run through a list of data structures and try to apply each one.
6. **Best Conceivable Runtime** (BCR)
   * The best conceivable runtie is, literally, the **best runtime** you could conceive of a solution to a problem. You can easily prove that there is no way you could beat the BCR.


## [Codeforces](https://codeforces.com/)
  * If you can achieve **1800 scores within 1 hour** that would be highly recommended for Google applicants.

## [Interviewbit](https://www.interviewbit.com/courses/programming/)
  * Level1:
    * Time Complexity
  * Level2:
    * Arrays
    * Math
  * Level3:
    * Binary Search
    * Strings
    * Bit Manipulation
    * Two Pointers
  * Level4:
    * Linked Lists
    * Stack and Queues
  * Level5:
    * Backtracking
    * Hashing
  * Level6:
    * Heaps And Maps
    * Tree
  * Level7:
    * Dynamic Programming
    * Greedy Algorithm
  * Level8
    * Graph Data Structure & Algorithms

## [Cracking the coding interview](http://www.crackingthecodinginterview.com/)

## [Leetcode](https://leetcode.com/)
### [Classification](https://cspiration.com/leetcodeClassification#103)
### Math
  * **Number**:
    * 007: Reverse Integer (E)
      * Notice the **boundary** and **negative** value
      * Approach1: Time:O(logn)
        * Python
            ```python
            def reverse(self, x: int) -> int:
              is_positive = True if x >=0 else False
              reverse = 0

              # 2 ** 31          = 2147483648
              # (2 ** 31) // 10  = 214748364
              boundary = (2 ** 31)//10 # 2147483640

              if not is_positive:
                  x = -x

              while x:
                  pop = x % 10
                  x //= 10
                  """
                  2 ** 31          = 2147483648
                  (2 ** 31) // 10  = 214748364 = boundary
                  2147483648 = 2147483640 + 8 = boundary * 10 + 8
                  """
                  if reverse > boundary \
                    or reverse == boundary and pop > 7 :
                      return 0

                  reverse = reverse * 10 + pop

              if not is_positive:
                  reverse = -reverse

              return reverse
            ```
    * 066: Plus One (E)
      * Approach1: Time:O(n), Space:O(1):
        * Python
          ```python
          def plusOne(self, digits: List[int]) -> List[int]:
            n = len(digits)
            carry = 1

            for i in range(n-1, -1, -1):
                val = carry + digits[i]
                digits[i] = val % 10
                carry = val // 10

            if carry:
                digits = [1] + digits

            return digits
          ```
    * 202: Happy Number (E)
        * Example:
          * end with 1:
            * 19 -> 82 -> 68 -> 100 -> **1** -> **1** ...
          * end without 1
            * 11 -> **4** -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> **4**
        * Approach1: Brute Force, Space:O(n)
          * Python
            ```python
            def isHappy(self, n: int) -> bool:
              def digit_square_sum(n):
                  s = 0
                  while n:
                      pop = n % 10
                      n = n // 10
                      s += pop ** 2
                  return s

              memo = dict()
              while True:
                  if n in memo:
                      break

                  memo[n] = True
                  n = digit_square_sum(n)

              return n == 1
            ```
        * Approach2: Detect Circle: Space:O(1)
          * Python
            ```python
            def isHappy(self, n: int) -> bool:
              def digit_square_sum(n):
                  s = 0
                  while n:
                      pop = n % 10
                      n = n // 10
                      s += pop ** 2
                  return s

              slow = fast = n
              while True:
                  slow = digit_square_sum(slow)
                  fast = digit_square_sum(digit_square_sum(fast))

                  if slow == fast:
                      break

              return slow == 1
            ```
    * Excel Sheet
      * 168: **Excel Sheet** Column Title (E)
        * Ref:
          * https://leetcode.com/problems/excel-sheet-column-title/discuss/51404/Python-solution-with-explanation
        * Example:
          ```txt
          A   1     AA    26+ 1     BA  2×26+ 1     ...     ZA  26×26+ 1     AAA  1×26²+1×26+ 1
          B   2     AB    26+ 2     BB  2×26+ 2     ...     ZB  26×26+ 2     AAB  1×26²+1×26+ 2
          .   .     ..    .....     ..  .......     ...     ..  ........     ...  .............
          .   .     ..    .....     ..  .......     ...     ..  ........     ...  .............
          .   .     ..    .....     ..  .......     ...     ..  ........     ...  .............
          Z  26     AZ    26+26     BZ  2×26+26     ...     ZZ  26×26+26     AAZ  1×26²+1×26+26

          ABCD＝A×26³＋B×26²＋C×26¹＋D＝1×26³＋2×26²＋3×26¹＋4
          WXYZ＝W×26³＋X×26²＋ZY26¹＋Z, W,X,Y,Z are in range 1~26
          ```
        * Approach1: Time:O(log(n)), Space:O(1)
          * Time:
            * Total O(log(n)) round
          * Space:
            * Each round will increase one character in the deque
          * Python:
            ```python
            def convertToTitle(self, n):
              title = collections.deque()
              ord_a = ord('A')

              while n > 0:
                  # n-1 to get number between 0(A) ~ 25(Z)
                  pop = (n-1) % 26
                  # n-1 to cut the final number completely
                  n = (n-1) // 26
                  title.appendleft(chr(ord_a + pop))

              return "".join(title)
            ```
        * Approach2: Time:O(log(n)), Space:O(k)
          * Python
            ```python
            def convertToTitle(self, n: int) -> str:
              title = collections.deque()
              memo = tuple(chr(ord_) for ord_ in range(ord('A'), ord('Z')+1))

              while n:
                  pop = (n - 1) % 26
                  n = (n - 1) // 26
                  title.appendleft(memo[pop])

              return "".join(title)

            ```
      * 171: **Excel Sheet** Column Number (E)
        * Approach1: Time:O(n), Space:O(1)
          * Python
            ```python
            def titleToNumber(self, s):
              ord_a = ord('A')
              result = 0

              for i in range(len(s)):
                  """
                  A -> 1
                  B -> 2
                  ..
                  Z -> 26
                  """
                  pop = ord(s[i]) - ord_a + 1
                  result = result * 26 + pop

              return result
            ```
    * Romain
      * 013: **Romain** to Integer (E)
        * Approach1: Scan from right to left and keep cur max, Time:O(n), Space:O(1)
          * Python
          ```python
          def romanToInt(self, s):
            if len(s) < 1:
                return 0

            scores = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
            cur_max = 0

            # n-1
            num = cur_max = scores[s[len(s)-1]]

            # from n-2 to 0
            for i in range(len(s)-2, -1, -1):
                symbol = s[i]
                score = scores[symbol]
                if score >= cur_max:
                    cur_max = max(cur_max, score)
                    num += score
                else:
                    num -= score

            return num
          ```
      * 012: Integer to **Roman** (M)
        * Approach1
          * Time: O(c)
          * Space: O(c)
          * FAQ:
            * Why should I create a list like
            ```python
            symbols = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
            values = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
            ```
          * Python
            ```python
            def intToRoman(self, num):
              symbols = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
              values = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
              res = ""
              while num:
                  for symbol, val in zip(symbols, values):
                      cnt = num//val  # symbol cnt
                      if not cnt:
                          continue
                      res += cnt * symbol
                      num %= val

              return res
            ```
    * Strobogrammatic
      * Definition:
        * A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
        * example:
          * 69
          * 88
          * 00
          * 11
      * 246:Strobogrammatic Number (E)
        * Approach1, Time:O(n), Space:O(1)
          * Python
            ```python
            def isStrobogrammatic(self, num):
              d = {'6':'9', '9':'6', '8':'8', '1':'1', '0':'0'}
              start, end = 0, str(num) - 1
              ret = True

              while left <= right:
                  # if left == right:
                  # the valid cases would be '8':'8', '1':'1', '0':'0'
                  c = num[left]
                  if c not in d or d[c != num[right]:
                      ret = False
                      break

                  left -= 1
                  right += 1

              return ret
            ```
      * 247 Strobogrammatic Number II (M)
        * Find all strobogrammatic numbers that are of length = n.
        * Approach1, Bottom up iterative: Time: O(nk^n), Space:O(nk^n)
          * Time: O(nk^n)
            * About O(k^n) combinations
            * each combination cost O(n) to copy
          * Space: O(nk^n)
            * About O(k^n) combinations
            * each combination needs O(n) space
          * Python:
            ```python
            def findStrobogrammatic(self, n: int) -> List[str]:
              if n < 1:
                  return []

              d_non_final = {'6':'9', '9':'6', '8':'8', '1':'1', '0':'0'}
              d_final = {'6':'9', '9':'6', '8':'8', '1':'1'}

              if n % 2:
                  # odd
                  combs = ["0" , "1", "8"]
              else:
                  combs = [""]

              for i in range(n//2):

                  d = d_non_final

                  if i == n//2-1:
                      d = d_final

                  next_combs = []
                  for s in combs:
                      for l, r in d.items():
                          next_combs.append(f'{l}{s}{r}')

                  combs = next_combs

              return combs
            ```
      * 248 Strobogrammatic Number II (H)
        * Write a function to count the total strobogrammatic numbers that exist in the range of low <= num <= high.
    * 273: Integer to **English Words** (H)
    * 065: Valid Number (H)
  * **Sum**
    * 001: Two Sum (E)
      * Approach1: hash table, Time: O(n), Space: O(n)
        * Python
          ```python
          def twoSum(self, nums: List[int], target: int) -> List[int]:
            """
            Assume exactly one solution
            """
            res = []
            d = dict()
            for idx, num in enumerate(nums):
                diff = target - num
                if diff in d:
                    res += [d[diff], idx]
                    break

                d[num] = idx

            return res
          ```
      * Find all available 2 sums: O(nlog(n))
        * O(nlogn)
          * Sorting
          * Left pointer and right pointer to find sum of left + right == target
        * Python
          ```python
          def two_sum(l, r, target, cur):
            while l < r:
                s = nums[l] + nums[r]
                if s == target:

                    results.append(cur + [nums[l], nums[r]])

                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1

                elif s < target:
                    l += 1
                else: # s > target
                    r -= 1

          results = []
          cur = []
          nums.sort()
          two_sum(0, len(nums)-1, target, cur)
          return results
          ```
    * 167: Two Sum II - Input array is sorted (E)
      * Approach1: Hash Table, See 001, Time:O(n), Space:O(n)
      * Approach2: Two pointers, Time:O(n), Space:O(n)
        * Python
          ```python
          def twoSum(self, numbers: List[int], target: int) -> List[int]:
            n = len(numbers)
            if n < 2:
                return []

            res = []
            l = 0
            r = n - 1

            while l < r:
                s = numbers[l] + numbers[r]
                if s == target:
                    res += [l+1, r+1]
                    break
                elif s < target:
                    l += 1
                else:
                    r -= 1

            return res
          ```
    * 1099: Two Sum Less Than K (E)
      * Approach1: Sort, Time:O(nlogn), Space:O(n)
        * Python
          ```python
          def twoSumLessThanK(self, A: List[int], K: int) -> int:
            nums, target = A, K

            nums.sort()
            l, r = 0, len(nums) - 1
            res = -1

            while l < r:
                s = nums[l] + nums[r]

                if s < target:
                    res = max(res, s)
                    l += 1
                else:  # s >= target
                    r -= 1

            return res
          ```
    * 015: 3Sum (M)
      * Approach1: Sort and find, Time: O(n^2), Space:O(sorting)
        * Time: O(n^2)
          * **Sort**
            * O(nlog(n))
          * For each target (from 0 to n-3): O(n^2)
               * Find left and right pair that num[target] + num[left] + num[right] = 0
               * Target = -num[target] = num[left] + num[right]
        * Why does the algorithm work?
            * Assume that we find a correct target for num[left] + num[right]
              * case1: x
                * nums[left+1] + nums[right+1] > Target
              * case2: x
                * nums[left-1] + nums[right-1] < Target.
              * case3: ✓
                * nums[left+1] + nums[right-1] may be possible.
              * case4: x
                * nums[left-1] + nums[r+1] has been traverse before.
        * Python Solution
            ```python
            def threeSum(self, nums: List[int]) -> List[List[int]]:
              def two_sum(l, r, target, cur):
                  while l < r:
                      s = nums[l] + nums[r]
                      if s == target:
                          results.append(cur + [nums[l], nums[r]])

                          while l < r and nums[l] == nums[l+1]:
                              l += 1
                          while l < r and nums[r] == nums[r-1]:
                              r -= 1

                          l += 1
                          r -= 1

                      elif s < target:
                          l += 1
                      else: # s > target
                          r -= 1

              n = len(nums)
              nums.sort()
              results, cur = [], []
              for i in range(n-2):

                if i > 0 and nums[i] == nums[i-1]:
                  continue

                cur.append(nums[i])
                two_sum(l=i+1,
                        r=n-1,
                        target=0-nums[i],
                        cur=cur)
                cur.pop()

              return results
            ```
    * 018: 4Sum (M)
      * Time Complexity:
        * 2sum : O(n)
        * 3sum : O(n^2), O(n) * O(2sum)
        * 4sum : O(n^3), O(n) * O(3sum)
        * ksum : O(n^(k-1)), O(n() * O((k-1)sum)
      * Approach1: Base on 3Sum, Time:O(n^3), Space:O(sorting)
        * Python
          ```python
          def fourSum(self, nums, target):
            results = []
            nums.sort()
            for i in range(len(nums)-3):
              if i == 0 or nums[i] != nums[i-1]:
                threeResult = self.threeSum(nums[i+1:], target-nums[i])
                for item in threeResult:
                  results.append([nums[i]] + item)
            return results
          ```
      * Approach2: kSum, Time:O(n^(k-1))
        * Ref:
          * https://leetcode.com/problems/4sum/discuss/8545/Python-140ms-beats-100-and-works-for-N-sum-(Ngreater2)
        * for k sum, notice the dupliate handling:
            ```python
            if i > l and nums[i] == nums[i-1]:
              continue
            ```
        * Python
          ```python
          def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
            def two_sum(l, r, target, cur):
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(cur + [nums[l], nums[r]])
                        while l < r and nums[l] == nums[l+1]:
                            l += 1
                        while l < r and nums[r] == nums[r-1]:
                            r -= 1
                        l += 1
                        r -= 1
                    elif s < target:
                        l += 1
                    else: # s > target
                        r -= 1

            def k_sum(l, r, k, target, cur):
                cnt = r - l + 1
                if k > cnt:
                    return

                if k == 2:
                    two_sum(l, r, target, cur)

                else: # k > 2
                    # (r+1) - (k-1) = r + 2 - k
                    # boundary: r+1 (inclusive)
                    # remain: k-1
                    for i in range(l, r+2-k):
                      if i > l and nums[i] == nums[i-1]:
                        continue

                      cur.append(nums[i])
                      # get k-1 sum in the nums[i+1:r+1]
                      k_sum(l=i+1,
                            r=r,
                            k=k-1,
                            target=target-nums[i],
                            cur=cur)
                      cur.pop()

            nums.sort()
            cur, results = [], []
            k_sum(l=0,
                  r=len(nums)-1,
                  k=4,
                  target=target,
                  cur=cur)
            return results
          ```
  * **Majority**
    * Definition:
      * The majority element is **the element that appears more than ⌊ n/2 ⌋ times**.
    * 1150: Check If a Number Is Majority Element in a Sorted Array (E)
      * This question is similar to 034: Find **First and Last Position of Element** in Sorted Array
      * Example:
          ```txt
          [ x x x x x x x . . . . . . ]   # majority at the beginning
          [ . . . x x x x x x x . . . ]   # majority at the middle
          [ . . . . . . x x x x x x x ]   # majority at the ending
          ```
      * Approach1: Binary Search, Time:O(logn)
        * Python
          ```python
          def isMajorityElement(self, nums: List[int], target: int) -> bool:

            def binary_search_left(left, right, target):
                while left <= right:
                    mid = (left + right) // 2
                    if target <= nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1

                return left

            def binary_search_right(left, right, target):
                while left <= right:
                    mid = (left + right) // 2
                    if target >= nums[mid]:
                        left = mid + 1
                    else:
                        right = mid - 1

                return right


            n = len(nums)
            m_left = binary_search_left(0, n-1, target)

            if m_left >= n-1 or nums[m_left] != target:
                return False

            m_right = binary_search_right(m_left, n-1, target)

            return (m_right - m_left + 1) > (n // 2)

          ```
    * 169: Majority Element (E)
      * Notice the odd and even cases
      * Approach1: Sorting, Time:O(nlogn), Space:O(sorting)
        * Python
        ```python
          def majorityElement(self, nums: List[int]) -> int:
            nums.sort()
            n = len(nums)
            return nums[n//2]
        ```
      * Approach2: Hash, Time:O(n), Space:O(n)
        * Python
          ```python
          def majorityElement(self, nums: List[int]) -> int:
            d = collections.defaultdict(int)
            threshold = len(nums) // 2

            for n in nums:
                d[n] += 1

                if d[n] >= threshold:
                    majority = n
                    break

            return majority
          ```
      * Approach3: Boyer-Moore Voting Algorithm, Time:O(n), Space:O(1)
        * Ref:
          * https://gregable.com/2013/10/majority-vote-algorithm-find-majority.html
        * Algo:
          * We maintain a count,
            * which is incremented whenever we see an instance of our current candidate for majority element
            * and decremented whenever we see anything else. Whenever count equals 0, we effectively forget about everything in nums up to the current index and consider the current number as the candidate for majority element.
        * Example:
          * [5,5,5,5,5,1,2,3,4] -> final cnt = 1, majority=5
          * [5,1,5,2,5,3,5,4,5] -> final cnt = 1, majority=5
          * [1,2,3,4,5,5,5,5,5] -> final cnt = 5, majority=5
          * After scanning the array,
            * The cnt of the majority should be greater than 1
            * The cnt of the non-majority should be equal to 0 and will be forgot.
        * Python
          * Implementation1, this implementation can be extended to k majorities.
            ```python
            def majorityElement(self, nums: List[int]) -> int:
              majority = None
              cnt = 0
              for n in nums:
                  if n == majority:
                      cnt += 1
                  elif cnt == 0:
                      majority = n
                      cnt = 1
                  else:
                      cnt -= 1

              return majority
            ```
          * Implementation2
            ```python
            def majorityElement(self, nums: List[int]) -> int:
              cnt = 0
              majority = None
              for n in nums:
                  if cnt == 0:
                      majority = n

                  if n == majority:
                      cnt += 1
                  else:
                      cnt -= 1

              return majority
            ```
    * 229: Majority Element II (M)
      * Definition
        * Given an integer array of size n, find all elements that appear more than **n/3** times.
      * Ref:
        * Boyer-Moore Voting Algorithm
          * https://gregable.com/2013/10/majority-vote-algorithm-find-majority.html
          * https://leetcode.com/problems/majority-element-ii/discuss/63537/My-understanding-of-Boyer-Moore-Majority-Vote
      * Approach1: Hash, Time:O(n), Space:O(n)
        * Python
          ```python
          def majorityElement(self, nums: List[int]) -> List[int]:
            d = collections.defaultdict(int)
            threshold = len(nums) // 3
            majorities = dict()

            for n in nums:
                if n in majorities:
                    continue

                d[n] += 1
                if d[n] > threshold:
                    majorities[n] = True

            return majorities.keys()
          ```
      * Approach2: Boyer-Moore Voting Algorithm, Time:O(n), Space:O(1)
        * Ref:
          * https://leetcode.com/problems/majority-element-ii/discuss/63520/Boyer-Moore-Majority-Vote-algorithm-and-my-elaboration
        *  Since the requirement is finding the majority for more than ceiling of [n/3], the answer would **be less than or equal to two numbers**.
        * **Pair has 3 elements** in this question
        * Python
          ```python
          def majorityElement(self, nums: List[int]) -> List[int]:
            """
            Since the requirement is finding the majority for more than ceiling of [n/3],
            the answer would **be less than or equal to two numbers
            """
            c1 = c2 = 0
            cnt1 = cnt2 = 0
            threshold = len(nums) // 3

            for n in nums:
                # notice the condition sequence
                if n == c1:
                    cnt1 += 1

                elif n == c2:
                    cnt2 += 1

                elif cnt1 == 0:
                    c1, cnt1 = n, 1

                elif cnt2 == 0:
                    c2, cnt2 = n, 1

                # pair (c1, c2, n)
                # filtering a set of three elements out
                else:
                    cnt1, cnt2 = cnt1-1, cnt2-1

            # check
            cnt1 = cnt2 = 0
            for n in nums:
                if n == c1:
                    cnt1 += 1
                elif n == c2:
                    cnt2 += 1

            majorities = []
            if cnt1 > threshold:
                majorities.append(c1)
            if cnt2 > threshold:
                majorities.append(c2)

            return majorities
          ```
  * **Pascal's Triangle**
    * 118: Pascal's Triangle (E)
      * Approach1: DP: Time:O(n^2), Space:O(n^2)
        * Python
          ```python
          def generate(self, numRows: int) -> List[List[int]]:
            if numRows == 0:
                return []

            if numRows == 1:
                return [[1]]

            # layer1
            pascal = [[1]]
            # layer2 to layern
            for layer in range(2, numRows+1):
                new_layer = [1] * layer
                prev_layer = pascal[-1]

                for i in range(1, layer-1):
                    new_layer[i] = prev_layer[i-1] + prev_layer[i]

                pascal.append(new_layer)

            return pascal
          ```
    * 119: Pascal's Triangle II (E)
      * Get kth Pascal index
      * Approach1: From end to beginning: Time:O(n^2), Space:O(k)
        * Python
          ```python
          def getRow(self, rowIndex: int) -> List[int]:
            if rowIndex == 0:
                return [1]

            pascal = [0] * (rowIndex + 1)
            pascal[0] = 1

            # layer 1 to rowIndex
            for _ in range(1, rowIndex+1):
                # from end to beginning
                for i in range(rowIndex, 0, -1):
                    pascal[i] += pascal[i-1]

            return pascal
          ```
### Sorting
  * Array Sorting:
    * 912: Sort an Array (M)
      * Approach1: **Insertion** Sort, Time:O(n^2), Space:O(1)
        * Insert 1th, 2th, ... n-1th elements to the array
        * Iterative: Time:O(n^2), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              # 1 to n-1
              for insert in range(1, len(nums)):
                  # insert-1 to 0
                  for i in range(insert-1, -1, -1):
                      if nums[i] <= nums[i+1]:
                          break

                      nums[i], nums[i+1] = nums[i+1], nums[i]

              return nums
            ```
        * Recursive: Time:O(n^2), Space:O(n)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              def insertion_sort(insert):
                  if insert < 1:
                      return

                  # backtrack
                  insertion_sort(insert-1)

                  for i in range(insert-1, -1, -1):
                      if nums[i] <= nums[i+1]:
                          break
                      nums[i], nums[i+1] = nums[i+1], nums[i]


              n = len(nums)

              if n < 2:
                  return nums

              insertion_sort(n-1)

              return nums
            ```
      * Approach2: **Selection** Sort, Time:O(n^2), Space:O(1)
        * Select min from 0 to n-2
        * Iterative: Time:O(n^2), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              n = len(nums)
              if n <= 1:
                  return nums

              # from 0 to n-2
              for select in range(0, n-1):
                  minimum = select
                  # from select + 1 to n-1
                  for compare in range(select+1, n):
                      if nums[compare] < nums[minimum]:
                          minimum = compare

                  if minimum != select:
                      nums[select], nums[minimum] = nums[minimum], nums[select]

              return nums
            ```
        * Recursive: Time:O(n^2), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              def selection_sort(select):
                  if select >= n-1:
                      return

                  # from select+1 to n-1
                  min_idx = select
                  # from select + 1 to n-1
                  for i in range(select+1, n):
                      if nums[i] < nums[min_idx]:
                          min_idx = i

                  if min_idx != select:
                      nums[select], nums[min_idx] = nums[min_idx], nums[select]

                  selection_sort(select+1)

              n = len(nums)

              if n < 2:
                  return nums

              selection_sort(select=0)

              return nums
            ```
      * Approach3: **Bubble** Sort, Time:O(n^2), Space:O(1)
        * Bubble max to the end from n-1 to 1
        * Iterative: Time:O(n^2), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              n = len(nums)

              if n < 2:
                  return nums

              # from n-1 to 1
              for bubble in range(n-1, 0, -1):
                  # from 0 to bubble-1
                  is_swap = False
                  for i in range(bubble):
                      if nums[i] > nums[i+1]:
                          nums[i], nums[i+1] = nums[i+1], nums[i]
                          is_swap = True

                  if not is_swap:
                      break

              return nums
            ```
        * Recursive: Time:O(n^2), Space:O(n)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              def bubble_sort(bubble):
                  if bubble < 1:
                      return

                  is_swap = False
                  for i in range(bubble):
                      if nums[i] > nums[i+1]:
                          nums[i], nums[i+1] = nums[i+1], nums[i]
                          is_swap = True

                  if not is_swap:
                      return

                  bubble_sort(bubble-1)


              n = len(nums)

              if n < 2:
                  return nums

              bubble_sort(n-1)

              return nums
            ```
      * Approach4: **Quick** Sort, Time:O(nlogn), Space:O(logn~n)
        * Iterative, Time:O(nlogn), Space:O(logn~n):
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              def get_partition(left, right):
                  pivot = right
                  ran = random.randint(left, right)
                  nums[ran], nums[pivot] = nums[pivot], nums[ran]
                  pivot_val = nums[pivot]

                  border = left
                  for cur in range(border, pivot):
                      if nums[cur] <= pivot_val:
                          nums[border], nums[cur] = nums[cur], nums[border]
                          border += 1

                  nums[border], nums[pivot] = nums[pivot], nums[border]

                  return border

              if len(nums) < 2:
                  return nums

              stack = []
              stack.append((0, len(nums)-1))
              while stack:
                  l, r = stack.pop()

                  p = get_partition(l, r)

                  # left partition
                  if l < p-1:
                      stack.append((l, p-1))

                  # right partition:
                  if p + 1 < r:
                      stack.append((p+1, r))

              return nums
            ```
        * Recursive, Time:O(nlogn), Space:O(logn~n):
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:

              def get_partition(left, right):
                  pivot = right
                  ran = random.randint(left, right)
                  nums[ran], nums[pivot] = nums[pivot], nums[ran]
                  pivot_val = nums[pivot]

                  border = left
                  for cur in range(border, pivot):
                      if nums[cur] <= pivot_val:
                          nums[border], nums[cur] = nums[cur], nums[border]
                          border += 1

                  nums[border], nums[pivot] = nums[pivot], nums[border]

                  return border


              def quick_sort(l, r):
                  if l >= r:
                      return

                  p = get_partition(l, r)
                  quick_sort(l, p-1)
                  quick_sort(p+1, r)


              quick_sort(0, len(nums)-1)

              return nums
            ```
      * Approach5: **Merge** Sort, Time:O(nlogn), Space:O(n)
        * Note:
          * Use indices to prevent extra copy
          * Copy subarrays before merge
        * Iterative
          * Python
            ```python
             def sortArray(self, nums: List[int]) -> List[int]:
              def merge_2_sorted_array(dst_start,
                                      left_start, left_end,
                                      right_start, right_end):

                  left = nums[left_start:left_end+1]
                  right = nums[right_start:right_end+1]

                  d = dst_start
                  l = r = 0
                  while l < len(left) and r < len(right):
                      if left[l] <= right[r]:
                          nums[d] = left[l]
                          l += 1
                      else:
                          nums[d] = right[r]
                          r += 1

                      d += 1

                  while l < len(left):
                      nums[d] = left[l]
                      l += 1
                      d += 1

              window_size = 1
              while window_size < len(nums):

                  left = 0
                  while left + window_size < len(nums):
                      mid = left + window_size
                      right = mid + window_size
                      merge_2_sorted_array(left,
                                           left, mid-1,
                                           mid, right-1)
                      left = right

                  window_size *= 2

              return nums
            ```
        * Recursive
          ```python
          def sortArray(self, nums: List[int]) -> List[int]:
              def merge_2_sorted_array(dst_start,
                                      left_start, left_end,
                                      right_start, right_end):

                  left = nums[left_start:left_end+1]
                  right = nums[right_start:right_end+1]

                  d = dst_start
                  l = r = 0

                  while l < len(left) and r < len(right):
                      if left[l] <= right[r]:
                          nums[d] = left[l]
                          l += 1
                      else:
                          nums[d] = right[r]
                          r += 1

                      d += 1

                  while l < len(left):
                      nums[d] = left[l]
                      l += 1
                      d += 1


              def merge_sort(left, right):
                  if left >= right:
                      return

                  mid = (left + right) // 2

                  merge_sort(left, mid)
                  merge_sort(mid+1, right)

                  merge_2_sorted_array(left,
                                      left, mid,
                                      mid+1, right)


              merge_sort(0, len(nums)-1)

              return nums
          ```
      * Approach6: **Heap** Sort, Time:O(nlogn), Space:O(1)
        * Note:
          * Parent to child
            * left child: 2n + 1
            * right child: 2n + 1
          * Child to parent
            * (n - 1) / 2
        * Algo
          * Step1:
            * heapify the array
          * Step2:
            * (n-1) rounds
              * swap the max to the end of the array, and heapify again
        * Iterative: Time:O(nlogn), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              def max_heapify(cur, boundary):
                  """
                  bubble down operation
                  """
                  while True:
                      left = cur * 2 + 1
                      right = cur * 2 + 2

                      cur_max = cur

                      if left < boundary and nums[left] >= nums[cur_max]:
                          cur_max = left

                      if right < boundary and nums[right] >= nums[cur_max]:
                          cur_max = right

                      # no need to swap or no children
                      if cur_max == cur:
                          break

                      nums[cur], nums[cur_max] = nums[cur_max], nums[cur]
                      cur = cur_max


              n = len(nums)
              # from n//2 to 0 (bottom up heapify)
              # takes O(n)
              for i in range(n//2, -1, -1):
                  max_heapify(cur=i, boundary=n)

              # from n-1 to 1
              for i in range(n-1, 0, -1):
                  nums[0], nums[i] = nums[i], nums[0]
                  max_heapify(cur=0, boundary=i)

              return nums
            ```
        * Recursive: Time:O(nlogn), Space:O(n)
          * Change for loop to recursive
          * Cheange max_heapify to recurisive
  * Linked List Sorting:
    * 147: Insertion Sort List (M)
    * 148: Sort list (M)
  * Others:
      * 280: Wiggle Sort (M)
         * Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]
         * Approach1: sorting, O(log(n))
           * Sort and then pair swapping
           * Python Solution
             ```python
             def wiggleSort(self, nums: List[int]) -> None:
              nums.sort()
              for i in range(1, len(nums)-1, 2):
                  nums[i], nums[i+1] = nums[i+1], nums[i]
             ```
         * Approach2: greedy, O(n)
           * Greedy from left to right
           * Python Solution
             ```python
             def wiggleSort(self, nums: List[int]) -> None:
              should_less = True
              for i in range(len(nums)-1):
                  if should_less:
                      if nums[i] > nums[i+1]:
                          nums[i], nums[i+1] = nums[i+1], nums[i]
                  else:
                      if nums[i] < nums[i+1]:
                          nums[i], nums[i+1] = nums[i+1], nums[i]

                  should_less = not should_less
             ```
      * 324: Wiggle Sort II (M)
        * Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]
        * Error answer:
          * test case:
            * [1,2,2,1,2,1,1,1,1,2,2,2]
          * Python
          ```python
          def wiggleSort(self, nums: List[int]) -> None:
            should_less = True
            for i in range(len(nums)-1):
                if should_less:
                    if nums[i] >= nums[i+1]:
                        nums[i], nums[i+1] = nums[i+1], nums[i]
                else:
                    if nums[i] <= nums[i+1]:
                        nums[i], nums[i+1] = nums[i+1], nums[i]

                should_less = not should_less

          ```
      * 075: Sort Colors (M)
         * Approach1: Quick sort, Time:O(nlog(n)), Space:O(log(n)~n)
         * Approach2: **Counting sort**, Time:O(n+k), Space:O(k)
           * Python:
           ```python
           def sortColors(self, nums: List[int]) -> None:
             """
             Do not return anything, modify nums in-place instead.
             """
             if not nums:
                 return

             # 3 colors only
             color_num = 3
             cnt_memo = [0] * color_num

             for num in nums:
                 cnt_memo[num] += 1

             p = 0
             for color, cnt in enumerate(cnt_memo):
                 for _ in range(cnt):
                     nums[p] = color
                     p += 1
           ```
         * Approach3: **Dutch National Flag Problem**, Time:O(n), Space:O(1)
           * Like 2 boundary quick sort
             * p0: boundary for 0
             * p2: boundary for 2
             * cur: runner
           * Notice the end condition
             * p0 points to the **next** position 0 can be put.
             * p2 points to the **next** position 2 can be put.
           * Python
             ```python
             def sortColors(self, nums: List[int]) -> None:

               if not nums:
                   return

               cur = p0 = 0
               p2 = len(nums)-1

               while cur <= p2:
                   if nums[cur] == 2:
                       nums[cur], nums[p2] = nums[p2], nums[cur]
                       p2 -= 1
                   elif nums[cur] == 0:
                       nums[cur], nums[p0] = nums[p0], nums[cur]
                       p0 += 1
                       # p0 only forwards 1, cur does not need to check again.
                       cur += 1
                   else:  # nums[cur] == 1
                       cur += 1
             ```
### String
  * Remove Duplicate
    * 316: Remove Duplicate Letters (H)
  * Encode and Decode:
    * 394: Decode String (M)
      * See Queue and Stack Section
    * 271: Encode and Decode Strings (M)
      * Approach1: Non-ASCII Delimiter
      * Approach2: Chunked Transfer Encoding
  * Text
    * 068: Text Justification (H)
  * **Edit Distance**
    * 161: One Edit Distance (M)
       * Notice the **zero edit distance cases**.
       * Approach1: Time O(n+m), Space (1):
         * Merge the insert and remove cases (find the short one)
         * Use short and long string pointers to traverse and compare
         * Python Solution
          ```python
          def isOneEditDistance(self, s: str, t: str) -> bool:
            """
            case1: Insert a character into s to get t
            case2: Delete a character from s to get t
            case3: Replace a character of s to get t
            """
            diff = len(s) - len(t)
            if abs(diff) >= 2:
                return False

            # find the short and long strings
            # case2 can be merged to case1
            same_len = False
            if diff < 0:
                short, long = s, t
            elif diff == 0:
                short, long = s, t
                same_len = True
            else:
                short, long = t, s

            # traverse the short one
            i = j = 0
            edit_cnt = 0
            while i < len(short):
                if short[i] == long[j]:
                    i += 1
                    j += 1
                else:
                    if edit_cnt:
                        return False
                    edit_cnt += 1

                    # This is case3
                    if same_len:
                        i += 1
                        j += 1
                    # This is case 1 and case2
                    else:
                        j += 1

            # zero edit distance case
            if same_len and not edit_cnt:
                return False

            # exactly one edit distance
            return True
          ```
    * 072: Edit Distance (H)
      * DP
  * **SubString**
     * 028: Implement **strStr** (E)
       * Find Sub-String
       * m is the length of txt and
       * n is the length of the pattern string.
       * Approach1: Brute Force, Time: O(mn), Space(1)
         * Python Solution
         ```python
         def strStr(self, haystack: str, needle: str) -> int:
            txt = haystack
            pattern = needle
            res = not_found = -1

            if txt == "" and pattern == "":
                return 0

            if txt and not pattern:
                return 0

            for i in range(len(txt)-len(pattern)+1):
                t = i
                p = 0
                while p < len(needle):
                    if txt[t] != pattern[p]:
                        break
                    t += 1
                    p += 1

                if p == len(pattern):
                    res = t - p
                    break

            return res
         ```
       * Approach2: **KMP (substring match)**, Time: O(m+n), Space: O(n)
         * Reference:
           * [Concept](https://www.youtube.com/watch?v=GTJr8OvyEVQ)
             * **Reuse the longest common prefix suffix for next pattern searching**.
               * **The current suffix range is the next prefix range**.
                 * please see the figure below
           * [The LPS table](http://jakeboxer.com/blog/2009/12/13/the-knuth-morris-pratt-algorithm-in-my-own-words/)
             * Definiton of **Proper Prefix** and **Proper Suffix**
               * For a pattern: "Snape"
                 * The **Proper Prefix** would be:
                   * S, Sn, Sna, Snap
                 * The **Proper Suffix** would be:
                   * nape, ape, pe, e
             * Definition of the value in prefix suffix table
               * **The length of the longest proper prefix** in the (sub)pattern that matches a proper suffix in the same (sub)pattern.
           * Example
             * ![KMP example](./image/algo/KMP.png)
               * When un-macth happends, **the max reusable string range for next round is in suffix range**.
               * Seach from LPS array and find the proper start position of pattern comparison pointer (in this example, index 3).
         * Python Solution
           ```python
           def get_lps(self, pattern):
            lps = [None] * len(pattern)
            lps[0] = 0
            p = 0  # prefix pointer
            s = 1  # suffix pointer

            while s < len(pattern):
                if pattern[s] == pattern[p]:
                    p += 1
                    lps[s] = p  # set index+1 of p to s
                    s += 1
                else:
                    if p:
                        # reuse the prefix string that has been scanned.
                        # The length of the longest common prefix suffix
                        # are put in lps[p-1]
                        p = lps[p-1]
                    else:
                        # do not match anything
                        lps[s] = 0
                        s += 1

           def strStr(self, haystack: str, needle: str) -> int:
            txt = haystack
            pattern = needle
            res = not_found = -1

            if txt == "" and pattern == "":
                return 0

            if txt and not pattern:
                return 0

            lps = self.get_lps(pattern)
            t = p = 0

            while t < len(txt):
                if txt[t] == pattern[p]:
                    t += 1
                    p += 1
                    if p == len(pattern):
                        res = t - p
                        break
                else:
                    if p:
                        # reuse the prefix string that has been scanned.
                        # The length of the longest common prefix suffix
                        # are put in lps[p-1]
                        p = lps[p-1]
                    else:
                        # do not match anything
                        t += 1

            return res
           ```
     * 003:	Longest Substring **Without Repeating** Characters (M)
       * Approach1: Brute Force, Time:O(n^2), Space:O(n)
         * Iterative all pairs and use hash table to keep previous values
           * [1], [1,2], [1,2,3], # start with index 0
           * [2], [2,3]           # start with index 1
           * [3]                  # start with index 2
         * Python Solution:
          ```python
          def lengthOfLongestSubstring(self, s: str) -> int:
              max_len = 0
              for start in range(len(s)):
                  d = dict()
                  cur_len = 0

                  for end in range(start, len(s)):
                      if s[end] in d:
                          break  # find duplicate

                      cur_len += 1
                      d[s[end]] = True
                      max_len = max(max_len, cur_len)

              return max_len
          ```
       * Approach2: Sliding Window, Time:O(n)~O(2n), Space:O(n)
         * Use hash table to keep  occurrence  of characters
         * For the window [start, end]
           * if s[end] is not in hash table.
             * Move end to end + 1 to extend window
             * Try to update maximum window len
           * else
             * Move start to start + 1 to move window
           **Note**:
             * End does not need to be move backward, since the window size will be smaller than the current max
             * So the key concept is how to forward the start.
         * Worst case: O(2n)
           * [a,a,a,a,a,a,a]
           * Each character need to be visited twice by start and end.
         * Python Solution:
           ```python
           def lengthOfLongestSubstring(self, s: str) -> int:
              max_len = start = end = 0
              d = dict()
              while end < len(s):
                  if s[end] not in d:
                      d[s[end]] = True
                      max_len = max(end-start+1, max_len)
                      end += 1         # forward end to extend window size
                  else:
                      d.pop(s[start])  # key in dict is unique
                      start += 1       # forward start to move window

              return max_len
           ```
       * Approach3: Sliding Window Optimized, Time:O(n), Space:O(n)
         * All combinatinons are:
           * start with index_0
           * start with index_1
           * ...
           * start with index_n-1
         * For the window [start, end]
           * Use hash table to keep index of each character.
           * if s[end] is not in hash table.
             * Move end to end + 1 to extend window
           * else
             * Move start to hash[s[end]] + 1 to move window
               * skip from start + 1 to s[end]
             * Example: ***
                ```txt
                idx 0  1  2  3  4  5  6
                val a  x  b  x  x  x  b
                    s  s1 s2 s3       end
                s should jump to s3.
                s1, s2 can be skipped, since their max window size would be
                smaller than the [s:end]
                ```
          * **Note**:
             * Dnd does not need to be move backwarded, since the window size will be smaller than the current max
             * So the key concept is how to forward the start.
          Python Solution:
            ```python
            def lengthOfLongestSubstring(self, s: str) -> int:
              max_len = start = 0
              d = dict()

              for end, c in enumerate(s):
                  # backward does not make sense
                  # for example: axxbxxbxxa, backward to first a cause error
                  if c in d and d[c] >= start:
                      start = d[c] + 1

                  max_len = max(max_len, end-start+1)
                  d[c] = end

              return max_len
            ```
     * 076: Minimum Window Substring (H)
       * Approach1: Brute Force: Time:O(m^2+n), Space:O(m+n)
         * Time:
           * Generate Hash Table pattern: O(n)
           * Compare all pairs : O(m^2)
         * Find all paris
           * [1], [1,2], [1,2,3], # start with index 0
           * [2], [2,3]           # start with index 1
           * [3]                  # start with index 2
         * Python Solution:
           ```python
           def minWindow(self, s, t):
            if not t or not s:
                return ""

            # substring length, start and end
            res = [float('inf'), None, None]
            counter_t = collections.Counter(t)
            desired_formed = len(counter_t)

            for start in range(len(s)):
                formed = 0
                counter_s = collections.defaultdict(int)
                for end in range(start, len(s)):
                    c = s[end]

                    counter_s[c] += 1
                    if counter_s[c] == counter_t[c]:
                        formed += 1

                    if formed == desired_formed:
                        w_len = end - start + 1
                        # print(start, end, s[start:end+1])
                        if w_len < res[0]:
                            res[0], res[1], res[2] = w_len, start, end

                        break # break since we want to find the minimum one

             return "" if res[0] == float('inf') else s[res[1]: res[2]+1]
           ```
       * Approach2: Sliding Window: Time:O(m+n), Space:O(m)
         * Time: O(m+n)
           * Generate Hash Table for Pattern: O(n)
           * Scan Txt string: O(m)
         * Space: O(m+n)
           * Hash Table for Pattern: O(m)
           * Hash Table for txt: O(m)
         * Ref:
           * https://leetcode.com/articles/minimum-window-substring/
         * Algo:
           * All combinatinons are:
             * start with index_0
             * start with index_1
             * ...
             * start with index_n-1
           * Steps:
             * s1: We start with two pointers, start and end initially pointing to the first element of the string s.
             * s2: We use the end pointer to expand the window until we get a desirable window i.e. a window that contains all of the characters of t.
               * **This is the minimum window starting with current start**
             * s3: Once we have a window with all the characters, we can move the left pointer ahead one by one. If the window is still a desirable one we keep on updating the minimum window size.
               * **Try to find other minimum window with other start**
             * s4: If the window is not desirable any more, we repeat step2 onwards.
           * example for step3:
              ```txt
              idx 0  1  2  3  4  5
              val x  x  x  x  x  x
                  s1 s2          end

              * Assume the minimum desired window starting from s1 is s[s1:end+1], if [s2:end+1] is also a desired window, then [s2:end+1] is the minimum window starting with s2
                * Proof: If [s2:end+1] is not, then [s1:end+1] will not be the minimum one.
              ```
         * Python
            ```python
            def minWindow(self, s, t):
              if not t or not s:
                  return ""

              # substring length, start and end
              res = [float('inf'), None, None]
              counter_t = collections.Counter(t)

              """
              formed is used to keep track of how many unique characters in t are present in the
              current window in its desired frequency.
              e.g. if t is "AABC" then the window must have two A's, one B and one C.
                  Thus formed would be = 3 when all these conditions are met.
              """
              cur_formed = 0
              desired_formed = len(counter_t)
              counter_s = collections.defaultdict(int)
              start = end = 0

              while end < len(s):
                  c = s[end]

                  if c in counter_t:
                      counter_s[c] += 1
                      if counter_s[c] == counter_t[c]:
                          cur_formed += 1

                  # start == end means len == 1
                  while start <= end and cur_formed == desired_formed:
                      c_start = s[start]
                      w_len = end - start + 1
                      if w_len < res[0]:
                          res[0], res[1], res[2] = w_len, start, end

                      if c_start in counter_t:
                          counter_s[c_start] -= 1
                          if counter_s[c_start] < counter_t[c_start]:
                              cur_formed -= 1
                      start += 1

                  end += 1


              return "" if res[0] == float('inf') else s[res[1]: res[2]+1]
            ```
     * 030: Substring with Concatenation of All Words (H)
     * 395: Longest Substring with **At Least K Repeating** Characters (M)
     * 340: Longest Substring with At Most K Distinct Characters (H)
     * 159: Longest Substring with At Most Two Distinct Characters (H)
  * **Palindrome**
     * 009: Palindrome **Number** (E)
       * Approach1: Covert to string, Time:O(n), Space:O(n)
         * Python Solution
          ```python
          def isPalindrome(self, x: int) -> bool:
            s = str(x)
            left, right = 0, len(s)-1

            res = True
            while left < right:
                if s[left] != s[right]:
                    res = False
                    break

                left += 1
                right -=1

            return res
          ```
       * Approach2: Reverse the interger, Time:O(log(n)), Space:O(1)
         * Python Solution
            ```python
            def isPalindrome(self, x: int) -> bool:
              if x < 0:
                  return False

              ref_x = x

              rev = 0
              while x:
                  pop = x % 10
                  x = x // 10
                  rev = rev * 10 + pop

              return rev == ref_x
            ```
     * 125:	**Valid** Palindrome (E)
       * Approach1: Time O(n), Space O(1)
         * Python Solution
          ```python
          def isPalindrome(self, s: str) -> bool:
              l = 0
              r = len(s)-1
              while l < r:
                  while l < r and not s[l].isalnum():
                      l += 1
                  while l < r and not s[r].isalnum():
                      r -= 1

                  if l >= r:
                      break

                  if s[l].lower() != s[r].lower():
                      return False

                  l += 1
                  r -= 1
              return True
          ```
     * 266:	Palindrome **Permutation** (E)
       * Approach1: Time O(n), Space O(c)
         * Python Solution
            ```python
            def canPermutePalindrome(self, s: str) -> bool:
              d = collections.defaultdict(int)
              odd_cnt = 0
              for c in s:
                  d[c] += 1
                  if d[c] % 2 == 1:
                      odd_cnt += 1
                  else:
                      odd_cnt -= 1

              return odd_cnt <= 1
            ```
     * 267:	Palindrome **Permutation** II (M)
     * 005: **Longest** Palindromic Substring (M)
       * Ref:
         * https://leetcode.com/problems/longest-palindromic-substring/solution/
       * Approach1: Brute Force: O(n^3), Space: O(1)
         * Time: O(n^3):
           * List all Combination of substrings: O(n^2)
           * Verify each substring: O(n)
         * Python Solution:
            ```python
            def longestPalindrome(self, s: str) -> str:
              """
              (0, 0), (0, 1), (0, 2), (0, 3)
                      (1, 1), (1, 2), (1, 3)
                              (2, 2), (2, 3)
                                      (3, 3)
              """
              def is_palindrom(left, right):
                  while left < right:
                      if s[left] != s[right]:
                          return False

                      left += 1
                      right -= 1

                  return True

              l = r = 0
              # total combinations
              for left in range(len(s)):
                  for right in range(left, len(s)):
                      if (right-left) > (r-l) and is_palindrom(left, right):
                          l, r = left , right

              return s[l:r+1]
            ```
       * Approach2: DP (enhancement of Brute Force), Time: O(n^2), Space: O(n^2)
         * The concept is like brute force, but reuse previous palindrom comparison results
          * memo[i][j]:
           * means s[i:j+1] is a palindrom or not
         * Rules:
           * case1: For i == j:
             * memo(i,i) = true
           * case2: For i+1 == j
             * memo(i,i+1) = (s(i) == s(i+1))
           * case3: For j > i + 1
             * memo(i,j) =  memo(i+1, j-1) and (s(i) == s(i+1))
               * memo(i+1, j-1) is substring result
               * eg, abcdcba -> bcdcb
         * Python Solution
            ```python
            def longestPalindrome(self, s: str) -> str:
              if not s:
                  return ""

              l = r = 0
              n = len(s)

              memo = [[False for _ in range(n)] for _ in range(n)]

              # init case1 and case2:
              for i in range(n):
                  # case1
                  memo[i][i] = True

                  if i + 1 == n:
                      break

                  # case2
                  if s[i] == s[i+1]:
                      memo[i][i+1] = True
                      if 2 > r-l:
                          l, r = i, i + 1

              """
              (0, 0), (0, 1), (0, 2), (0, 3)
                      (1, 1), (1, 2), (1, 3)
                              (2, 2), (2, 3)
                                      (3, 3)
              """
              # from n-3 to 0, skip length <=2 (case1 and case2)
              for left in range(n-3, -1, -1):
                  for right in range(left+2, n):
                      # case3
                      if s[left] == s[right] and memo[left+1][right-1]:
                          memo[left][right] = True
                          if right-left > r-l:
                              l, r = left, right

              return s[l:r+1]
            ```
       * Approach3: Expand Center, Time: O(n^2), Space: O(1)
         * **Expand the center**, there are total 2n-1 center
           * odd case:
             * a**b**a -> the center is b
           * even case
             * a**bb**a -> the center is between bb
         * Python Solution
          ```python
          def longestPalindrome(self, s: str) -> str:

            def expand_center(left, right):
                while left > -1 and right < len(s):
                    if s[left] != s[right]:
                        break

                    left -= 1
                    right += 1

                # out of boundary or s[left] != s[right]
                return left+1, right-1

            if not s:
                return ""

            l = r = 0
            n = len(s)

            for center in range(n):
                l1, r1 = expand_center(left=center, right=center)
                if (r1 - l1) > (r - l):
                    l, r = l1, r1

                l2, r2 = expand_center(left=center, right=center+1)
                if (r2 - l2) > (r - l):
                    l, r = l2, r2

            return s[l:r+1]
            ```
       * Approach4: Manacher's Algorithm, Time: O(n), Space: O(n)
         * Ref:
           * https://www.youtube.com/watch?v=nbTSfrEfo6M
     * 214:	**Shortest** Palindrome (H)
       * The problem can be converted to longest palindrome substring **starts from 0**
         * For example: str = "abade"
             * The longest palindrom substr substring from 0 is "aba"
             * The **reverse** the substring after "aba" is "ed"
             * Append "ed" before the head, 'ed'abadede is the answer.
       * Approach1: Brute Force, Time:O(n^2), Space:O(n)
         * Reverse the string
         * Find the **longest common prefix of the str and suffix of reverse str**
         * For example:
           * str = "abade"
           * rev = "edaba"
           * The longest common suffix of rev and prefix of str is "aba"
           * so the result would be xx + abaxx = xxabaxx
         * Python Solution
            ```python
            def shortestPalindrome(self, s: str) -> str:
              rev = s[::-1]
              res = ""
              for i in range(0, len(s)):
                  # try to find longest common suffix of rev and prefix of s
                  if rev[i:] == s[:len(s)-i]:
                      res = rev[:i] + s
                      break
              return res
            ```
       * Approach2: KMP, Time:O(n), Space:O(n)
         * Use Kmp to speed up finding the **longest common prefix of the str and suffix of reverse str**
         * For example:
           * str = "abade"
           * rev = "edaba"
           * create a tmp string **str + # rev_str**
             * '#' is used to force the match in reverse str starts from its first index)
             ```txt
             a b a d e # e d a b a
             0 0 1 0 0 0 0 0 1 2 3 <--- 3 is what we want
             ```
          * we know the length of the palindrom substring is 3, final anser would be
            * ed + abade = edabade
          * Python Solution:
              ```python
              def shortestPalindrome(self, s: str) -> str:

                def get_lps(s):
                    lps = [0] * len(s)
                    i = 0 # pointer prefix
                    j = 1 # pointer to suffix
                    while j < len(s):
                        if s[j] == s[i]:
                            i += 1
                            lps[j] = i
                            j += 1
                        else:
                            if i:
                                i = lps[i-1]
                            else:
                                lps[j] = 0
                                j += 1
                    return lps

                if not s:
                    return ""

                if len(s) == 1:
                    return s

                # separator is used to force the match in reverse str starts from its first index
                separator = '#'
                rev = s[::-1]
                """
                string + separator + reverse string
                """
                lps = get_lps(f'{s}{separator}{rev}')
                palindrom_len = lps[-1]

                return rev[:len(s)-palindrom_len] + s
              ```
     * 336:	Palindrome **Pairs** (H)
       * n: number of words
       * k: average length of each word
       * Approach1: Brute force: Time:O(n^2*k), Space:O(1)
         * Iteraitve each pair and check
         * (0, 1), (0, 2), (0, 3)  # each pair need to check twice: (0,1) and (1,0)
         * (1, 2), (1, 3)
         * (2, 3)
       * Approach2: Use Trie
         * Corner Cases:
           * Two words shoud be distinct from each other, how to avoid ?
     * 131:	Palindrome **Partitioning** (M)
     * 132:	Palindrome **Partitioning** II (M)
  * **Parentheses**
     * 020: Valid Parentheses (E)
       * Valid Cases:
         * [] () {}
         * { [ () ] }
       * Invalid Cases:
         * { [ () ]
         * ))
         * ((
       * Approach1: Use stack, Time:O(n), Space:O(n)
         * Python Solution
           ```python
           BRACKET_MAP = {'(': ')', '[': ']', '{': '}'}
           def is_valid_parentheses(self, s: str) -> bool:
               if not s:
                   return True

               ret = True
               stack = list()
               for c in s:
                   if c in BRACKET_MAP:
                       stack.append(c)
                   else:
                       # can not find the left bracket
                       if not stack or c != BRACKET_MAP[stack.pop()]:
                           return False

               # if len(stack) !=0 means that can not find the right bracket
               return len(stack) == 0
           ```
     * 022: Generate Parentheses (M)
       * See backtracking
     * 241: Different Ways to Add Parentheses (M)
     * 032:	Longest Valid Parentheses (H)
     * 301: Remove Invalid Parentheses (H)
  * **Subsequence**
    * Definition:
      * A subsequence of a string is a new string which is **formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters.**
    * 392: Is Subsequence (E)
      * Approach1: Iterate pattern, Time:O(n), Space:O(1)
        * Python Solution
          ```python
          def isSubsequence(self, s: str, t: str) -> bool:
            if not s:
                return True

            i = 0
            res = True
            for c in s:
                while i < len(t) and t[i] != c:
                    i += 1

                if i == len(t):  # not found
                    res = False
                    break

                i += 1

            return res
          ```
      * Approach2: Iterate original string, Time:O(n), Space:O(1)
        * Python Solution
          ```python
            def isSubsequence(self, s: str, t: str) -> bool:
              if not s:
                  return True

              res = False
              i = j = 0
              for c in t:
                  if c == s[i]:
                      i += 1
                      if i == len(s):
                          res = True
                          break
              return res
          ```
    * Longest Common Subsequence (**LCS**) (DP)
    * 300: Longest Increasing Subsequence (**LIS**) (M)
      * Approach1: Top down Recursive, Time:O(n^2), Space:O(n)
        * 2 cases:
          * case1: taken the current character if possible (cur > prev)
          * case2: do not tkaen the current character
        * Python
          ```python
          def lengthOfLIS(self, nums: List[int]) -> int:
            def get_lis(prev, cur, lis):
                # end condition
                if cur == n:
                    return

                nonlocal max_lis

                # taken the cur character
                if prev == -1 or nums[cur] > nums[prev]:
                    taken_lis = 1 + lis
                    max_lis = max(max_lis, taken_lis)
                    get_lis(cur, cur+1, taken_lis)

                # not taken
                get_lis(prev, cur+1, lis)

            if not nums:
                return 0

            max_lis = 0

            n = len(nums)

            get_lis(prev=-1, cur=0, lis=0)

            return max_lis
          ```
      * Approach2: Bottom up iterative + memo, Time:O(n^2), Space:O(n)
        * Python
          ```python
          def lengthOfLIS(self, nums: List[int]) -> int:
            n = len(nums)

            if n == 0:
                return 0

            if n == 1:
                return 1

            memo = [0] * n
            memo[0] = 1
            max_lis = 1
            for cur in range(1, n):
                lis = 1
                for prev in range(0, cur):
                    if nums[cur] > nums[prev]:
                        lis = max(lis, memo[prev]+1)

                memo[cur] = lis
                max_lis = max(max_lis, lis)

            return max_lis
          ```
      * Approach3: Binary Search, Time:O(nlogn), Space:O(n)
        * Ref:
          * https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation
          * https://www.youtube.com/watch?v=S9oUiVYEq7E
        * Use binary search to either
          * 1. extend increasing sequence with larger numbers
            * if n is larger than all tails, append it, increase the size by 1
          * 2. minimize existing values with smaller ones - so we can use smaller numbers to extend it.
            * if x == tails[mid]
              * do nothing
            * else tails[i-1] < x < tails[i]
              * update tails[i] (the smallest val which is greater than x)
        * Boundaries:
          * left boundary:
            * val before left is smaller than the target
          * right boundary:
            * val after right is greater than the target
        * Python
          ```python
          def lengthOfLIS(self, nums: List[int]) -> int:
            tails = [0 for _ in range(len(nums))]
            size = 0

            for n in nums:
                left, right = 0, size-1
                need_update = True

                while left <= right:
                    mid = (left + right) // 2
                    # do not need to update
                    if n == tails[mid]:
                        need_update = False
                        break
                    elif n > tails[mid]:
                        left = mid + 1
                    else:  # n < tails
                        right = mid - 1

                # update the smallest val which is greater than x
                if need_update:
                    tails[left] = n
                    size = max(size, left + 1)

            return size
          ```
    * 187: Repeated DNA Sequences (M)
    * 115: Distinct Subsequences (H)
  * **Reorder**
     * 344: Reverse String (E)
       * Python Solution
       ```python
       def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        if len(s) <= 1:
            return

        left, right = 0, len(s)-1

        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
       ```
     * 541: Reverse String II (E)
       * You need to **reverse the first k characters for every 2k characters** counting from the start of the string.
         * example1
           * input: s = "abcdefg", k = 2
           * output: "bacdfeg"
         * example
           * input: s = "abcdefgh", k = 3
           * output: "cbadefhg"
       * Python Solution
        ```python
        def reverseStr(self, s: str, k: int) -> str:
          def reverse_list(left, right):
              while left < right:
                  l[left], l[right] = l[right], l[left]
                  left += 1
                  right -= 1

          if k <= 1 or not s:
              return s

          l = list(s)
          n = len(l)
          left = 0

          while left < n:
              right = min(left+k-1, n-1)
              reverse_list(left, right)

              left = left + 2*k

          return ''.join(l)
        ```
     * 151:	Reverse **Words** in a String	(M)
       * Approach1, from end to the end, Time O(n), Space O(n) (one pass)
         * From end to beginning of the string
         * Python solution
          ```python
          def reverseWords(self, s: str) -> str:
            w_end = len(s) - 1
            left_boundary = -1
            output = list()

            while w_end > left_boundary:
                while w_end-1 > left_boundary and s[w_end].isspace():
                    w_end -= 1

                # can not find the word
                if s[w_end].isspace():
                    break

                w_start = w_end
                while w_start-1 > left_boundary and not s[w_start-1].isspace():
                    w_start -= 1

                if output:
                    output.append(" ")

                output.append(s[start:w_end+1])

                w_end = start - 1

            return "".join(output)
          ```
       * Approach2, Time O(n), Space O(n)
         * Reverse the string, from left to right, reverse each word.
         * Python solution
          ```python
          def reverseWords(self, s: str) -> str:
            # 1. trim extra space
            # 2. transfer to list
            # 3. reverse the list
            l = list(" ".join(s.split()))[::-1]

            w_start = 0
            while w_start < len(l):
                while w_start+1 < len(l) and l[w_start].isspace():
                    w_start += 1

                # can not find the word
                if l[w_start].isspace():
                    break

                w_end = w_start
                while w_end+1 < len(l) and not l[w_end+1].isspace():
                    w_end +=1

                reverse_list(l, w_start, w_end)

                w_start = w_end + 1

            return "".join(l)
            ```
     * 186:	Reverse **Words** in a String II (M)
       * Approach1, Time:O(n), Space:O(1)
         * Python Solution
         ```python
         def reverseWords(self, s: List[str]) -> None:
          def reverse_list(left, right):
            while left < right:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1

          reverse_list(0, len(s)-1)

          w_start = 0
          boundary = len(s)
          while w_start < boundary:
              while w_start + 1 < boundary and s[w_start].isspace():
                  w_start += 1

              if s[w_start].isspace():
                  break

              w_end = w_start
              while w_end + 1 < boundary and not s[w_end+1].isspace():
                  w_end += 1

              reverse_list(w_start, w_end)
              # skip the space after w_end
              w_start = w_end + 2
         ```
     * 345:	Reverse **Vowels** of a String (E)
       * Approach1: Hash Table, Time:O(n), Space:O(k)
       * Python Solution
        ```python
          def reverseVowels(self, s: str) -> str:
            vowels = collections.Counter('aeiouAEIOU')
            l = list(s)
            start,end = 0, len(l) - 1

            while start < end:
                while start < end and l[start] not in vowels:
                    start += 1

                while start < end and l[end] not in vowels:
                    end -= 1

                if start >= end:
                    break

                l[start], l[end] = l[end], l[start]
                start +=1
                end -= 1

          return "".join(l)
        ```
     * 358: Rearrange String k Distance Apart (H)
  * **Isomorphism** and **Pattern**
     * 205: **Isomorphic** Strings (E)
        * Example:
          * aabbaa,  112233 -> False
          * aabbaa,  112211 -> True
        * Approach1, Use Hash Table
          * Use hash Table to keep **last seen index**
          * Python Solution
            ```python
            def isIsomorphic(self, s: str, t: str) -> bool:
                if len(s) != len(t):
                    return False

                # -1 means that this char does not appear before
                f = lambda: -1
                ds = collections.defaultdict(f)
                dt = collections.defaultdict(f)

                res = True
                for i in range(len(s)):
                    if ds[s[i]] != dt[t[i]]:
                        res = False
                        break

                    ds[s[i]] = dt[t[i]] = i
                return res
            ```
     * 290: Word **Pattern** (E)
       * The same concept as 205
       * Approach1, Use Hash Table
          * Use hash Table to keep **last seen index**
       * Python Solution
         ```python
         def wordPattern(self, pattern: str, str: str) -> bool:
          if not str or not pattern:
              return False

          txt = str.split()
          if len(pattern) != len(txt):
              return False

          f = lambda : -1
          d1 = collections.defaultdict(f)
          d2 = collections.defaultdict(f)

          res = True
          for i in range(len(pattern)):
              if d1[pattern[i]] != d2[txt[i]]:
                  res = False
                  break
              d1[pattern[i]] = d2[txt[i]] = i

          return res
         ```
  * **Anagram**
    * The key is how to calculate **signatures**.
     * 242: Valid Anagram (E)
       * Approach1: Use hash table, Time:O(n), Space:O(n)
         * Python Solution
          ```python
          def isAnagram(self, s: str, t: str) -> bool:
            if len(s) != len(t):
                return False

            d = collections.defaultdict(int)
            for c in s:
                d[c] += 1

            res = True
            for c in t:
                if d[c] == 0:
                    res = False
                    break
                d[c] -= 1

            return res
         ```
       * Approach2: Use sort, Time:O(nlogn), Space:O(n)
     * 049: Group Anagrams (M)
       * n is the number of strings, k is the maximum length of the strings
       * Approach1: Categorized by **sorted string**, Time: O(n*klog(k)) Space: O(nk)
         * Python Solution
            ```python
            def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
              d = collections.defaultdict(list)
              for s in strs:
                d[tuple(sorted(s))].append(s)
            return d.values()
            ```
       * Approach2: Categorized by **Character Count**, Time: O(nk), Space: O(nk)
         * Python Solution
            ```python
            def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
              def get_key(s):
                  count_arr = [0] * 26
                  for c in s:
                      count_arr[ord(c)-ord_base] += 1

                  return tuple(count_arr)

              d = collections.defaultdict(list)
              ord_base = ord('a')
              for s in strs:
                  k = get_key(s)
                  d[k].append(s)

              return d.values()
            ```
     * 249: Group **Shifted Strings** (M)
       * Time: O(nk)
         * n is the number of strings
         * k is the length of the string
       * Approach1: Group by diff string, Time:O(nk)
         * Example:
           * [a, b] and [z, a]
             * ord(a) = 97
             * ord(b) = 98
             * ord(z) = 122
           * (ord(b) - ord(a)) ≡ 1 (mod 26)
           * (ord(a) - ord(z)) ≡ -25 ≡ 1 (mod 26)
           * 1 is **congruent** to 25 (modulo 26)
       * Python Solution
         ```python
        def groupStrings(self, strings: List[str]) -> List[List[str]]:
          def get_key(s):
              ord_first = ord(s[0])
              return tuple((ord(c)- ord_first)%26  for c in s)

          d = collections.defaultdict(list)
          for s in strings:
              k = get_key(s)
              d[k].append(s)

          return d.values()
         ```
  * Deduction:
    * 038: Count and Say (E)
      * Approach1:
        * Python Solution
          ```python
          def countAndSay(self, n: int) -> str:
            # 1
            cur = "1"
            # 2 to n
            for _ in range(2, n+1):
                start = end = 0
                nxt = []
                while start < len(cur):
                    cnt = 0
                    while end < len(cur) and cur[end] == cur[start]:
                        cnt += 1
                        end += 1

                    nxt.append(f"{cnt}{cur[start]}")
                    start = end

                cur = "".join(nxt)

            return cur
          ```
    * 293: Flip Game (E)
        * python solution
        ```python
        def generatePossibleNextMoves(self, s: str) -> List[str]:
          output = list()
          # from 0 to n-2
          for i in range(len(s)-1):
              if s[i] == s[i+1] == '+':
                  output.append(f"{s[0:i]}--{s[i+2:]}")
          return output
        ```
    * 294: Flip Game II (M)
      * Approach1: backtracking: Time: O(n!!), Space: O(n*2)
        * **Double factorial**: (n-1) * (n-3) * (n-5) * ... 1=
         * python solution
           ```python
           def canWin(self, s: str) -> bool:
             # from 0 to len(s)-2
             for i in range(0, len(s)-1):
                 # the 1st makes the flip.
                 if s[i] == s[i+1] == '+':
                     first_flip_s = f"{s[0:i]}--{s[i+2:]}"

                     # the 2nd person makes the flip.
                     if not self.canWin(first_flip_s):
                         # 1st person wins the game
                         return True

             # can not make any flips or 2nd person always wins
             # this is end condition
             return False
           ```
      * Approach1: backtracking with memo
        * time complexity:
          * number_of_distinct_strings * each_unique_string_first_time_computation_contribution
            * O(2^n) * n (not sure)
        * space complexity:
          * O(n^2)
        * python solution
           ```python
           def canWin(self, s: str) -> bool:
             memo = dict()

             def _canWin(self, s):
                 if s in memo:
                     return memo[s]

                 # from 0 to len(s)-2
                 for i in range(0, len(s)-1):
                     if s[i] == s[i+1] == '+':
                         first_flip_s = f"{s[0:i]}--{s[i+2:]}"
                         # the 2nd flip
                         if not self.canWin(first_flip_s):
                             # first person wins the game
                             memo[s] = True
                             return True

                 # can not make any flips or 2nd person always wins
                 # this is end condition
                 memo[s] = False
                 return False

             return _canWin(self, s)
           ```
  * Other
     * 387: First Unique Character in a String (E)
       * Approach1: Use Hash Table to count the occurrence, Time: O(n), Space: O(c)
         * Python Solution
         ```python
         def firstUniqChar(self, s: str) -> int:
          res = not_found = -1
          counter = collections.Counter(s)
          for i, c in enumerate(s):
            if counter[c] == 1:
              res = i
              break

          return res
         ```
     * 058: Length of Last Word (E)
       * Approach1: Seach **from the end to the beginning**.
       * Python Solution
          ```python
          def lengthOfLastWord(self, s: str) -> int:
            w_count = 0
            found = False
            # traverse from end to start
            for i in range(len(s)-1, -1, -1):
                if s[i] != ' ':
                    w_count += 1
                    found = True
                else:
                    # we have found a word before
                    if find:
                        break

            return w_count
          ```
     * 014: Longest **Common Prefix** (E)
       * Approach1: **vertical scanning**, Time: O(mn), Space: O(1)
         * Notice the edge cases:
           * 1 strings
           * 1 character for each string, [a, b]
         * Time: O(mn)
           * Where m is the minimum length of str in strs and n is the number of strings.
         * Python Solution
             ```python
             def longestCommonPrefix(self, strs: List[str]) -> str:
              if not strs:
                return ""

              """
              Need to cover
              1. one string
              2. one character with multiple strings
              """
              # vertical scanning
              # prefix is the maximum length of the common prefix
              common_prefix = prefix = strs[0]
              leave = False

              # for each character in the prefix
              for idx, c in enumerate(prefix):
                  # compare with character in the same idx with other strings
                  for i in range(1, len(strs)):
                      compare_str = strs[i]
                      if idx < len(compare_str) and c == compare_str[idx]:
                        continue

                      common_prefix = prefix[0:idx]
                      leave = True
                      break

                  # break outer loop
                  if leave:
                      break

              return common_prefix
             ```
     * 383: Ransom Note (E)
        * Approach1: Hash Table, Time:O(m+n), Space:O(m)
          * Python Solution
          ```python
          def canConstruct(self, ransomNote: str, magazine: str) -> bool:
            d = collections.Counter(magazine)

            for w in ransomNote:
                if w not in d or d[w] <= 0:
                    return False
                else:
                    d[w] -= 1

            return True
          ```
     * 087: Scramble String (H)
### Array
  * **Check Duplicate**
    * 217: Contains Duplicate (E)
      * Approach1: Use hash Table
        * Python Solution
          ```python
          def containsDuplicate(self, nums: List[int]) -> bool:
            d = dict()
            res = False

            for num in nums:
                if num not in d:
                    d[num] = True
                else:
                    res = True
                    break

            return res
          ```
    * 219: Contains Duplicate II (E)
      * Find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.
      * Approach1: Use hash Table to store index.
        * Python Solution
        ```python
        def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
          if not nums:
              return False

          d = dict()
          res = False

          for idx, num in enumerate(nums):
              if num in d and idx - d[num] <= k:
                  res = True
                  break

              d[num] = idx

          return res
        ```
    * 287: Find the Duplicate Number (M)
      * Given an array nums **containing n + 1 integers** where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.
        * intergers range: 1 ~ n (n)
        * idx range: 0 ~ n (n+1)
      * Approach1: Sorting, Time:O(nlogn), Space:O(n)
        * Python
          ```python
          def findDuplicate(self, nums: List[int]) -> int:
            n = len(nums)

            if n <= 1:
                return None

            nums = sorted(nums)
            target = None
            for i in range(0, n-1):
                if nums[i] == nums[i+1]:
                    target = nums[i]
                    break

            return target
          ```
      * Approach2: Hash Table, Time:O(n), Space:O(n)
        * Python
          ```python
          def findDuplicate(self, nums: List[int]) -> int:
            if len(nums) <= 1:
                return None

            d = dict()
            target = None
            for n in nums:
                if n in d:
                    target = n
                    break

                d[n] = True

            return target
          ```
      * Approach3: Floyd's Tortoise and Hare (Cycle Detection), Time:O(n), Space:O(1)
        * The beginning of the loop if the duplicate number
        * Note
          * intergers range: 1 ~ n (n)
          * idx range: 0 ~ n (n+1)
          * Because each number in nums is between 1 and n, **it will necessarily point to an index that exists**.
            * **The val is also the idx of next val**
          * Example:
            * nums = [1, 2, 3, 2], n = 3
            * 1 -> 2 nums[1] -> 3 nums[2] -> 2 nums[3] -> 2 nums[1] -> ...
        * Python
          ```python
          def findDuplicate(self, nums):
            slow = fast = nums[0]

            while True:
                slow = nums[slow]
                fast = nums[nums[fast]]
                if slow == fast:
                    break

            target = nums[0]
            while True:
                # the intersection may be the first element
                if target == slow:
                    break
                target = nums[target]
                slow = nums[slow]

            return target
          ```
  * **Remove Duplicate**
    * 027: Remove elements (E)
      * Approach1:
        * Like partition step of quick sort (keep the border)
        * Copy the wanted elements to the position of the current border
        * Python Solution
          ```python
          def removeElement(self, nums: List[int], val: int) -> int:
            border = 0
            for num in nums:
                if num != val:
                    nums[border] = num
                    border += 1

            return border
          ```
    * 026: Remove **Duplicates** from **Sorted Array** (E)
      * Python Solution
        ```python
        def removeDuplicates(self, nums: List[int]) -> int:
          i = 0
          for j in range(1, len(nums)):
              if nums[j] != nums[i]:
                  i += 1
                  nums[i] = nums[j]
          return i + 1
        ```
    * 080: Remove **Duplicates** from **Sorted Array** II (M) *
      * Approach1: Use cnt
        * Python Solution:
          ```python
          def removeDuplicates(self, nums: List[int]) -> int:
            if not nums:
                return 0

            max_duplicate = 2
            i, cnt = 0, 1
            for j in range(1, len(nums)):
                if nums[i] == nums[j]:
                    cnt += 1
                    if cnt <= max_duplicate:
                        i += 1
                        nums[i] = nums[j]

                else:
                    i += 1
                    nums[i] = nums[j]
                    cnt = 1

            return i + 1
          ```
      * Approach2: See backward
        * Python Solution
          ```python
          def removeDuplicates(self, nums: List[int]) -> int:
            i = 0
            for n in nums:
                if i < 2 or n > nums[i-2]:
                    nums[i] = n
                    i += 1
            return i
          ```
  * **Containers**
    * 011: Container With Most Water (M)
      * How to calculate the area?
        * min(left_border, right_border) * width
      * Approach1: brute force, Time: O(n^2), Space: O(1)
        * Calculating area for all height pairs.
        * Python Solution:
        ```python
        def maxArea(self, height: List[int]) -> int:
          if not height:
              return 0

          max_area = 0
          n = len(height)

          # (0, 1), (0, 2), (0, 3)
          # (1, 2), (1, 3)
          # (2, 3)
          for left in range(0, n-1):
              for right in range(left+1, n):
                  h = min(height[left], height[right])
                  w = right - left
                  area = h * w
                  max_area = max(max_area, area)

          return max_area
        ```
      * Approach2: greedy with 2 pointers, Time: O(n), Space: O(1)
          * Move the index with shorter height to find the bigger area.
          * Python Solution
            ```python
            def maxArea(self, height: List[int]) -> int:
              if not height:
                  return 0

              max_area = 0
              left, right = 0, len(height)-1  # start from max width

              while left < right:
                  h = min(height[left], height[right])
                  w = right - left
                  area = h * w
                  max_area = max(max_area, area)

                  # move the shorter border
                  if height[left] <= height[right]:
                      left += 1
                  else:
                      right -= 1

              return max_area
            ```
    * 042: Trapping Rain Water (H) *
      * Ref:
        * https://leetcode.com/problems/trapping-rain-water/solution/
      * How to calculate the area?
        * Sum water amount of **each bin** (width=1)
          * 1. Find left border
          * 2. Find right border
          * Water area in the ith bin would be:
            * **min(left_border, right_border) - height of ith bin**
      * Approach1: DP, Time: O(n), Space: O(n)
        * Keep two arrays
          * left_max
            * The left_max[i] is the **left border** in ith bin.
          * right_max
            * The right_max[i] is the **right border** in ith bin.
          * Python Solution
            ```python
            def trap(self, height: List[int]) -> int:
              if not height or len(height) <= 2:
                return 0

              n = len(height)

              left_max = [0] * n
              right_max = [0] * n
              left_max[0] = height[0]
              right_max[n-1] = height[n-1]


              for i in range(1, n):
                  """
                  left_max[i] should consider height[i]
                  since if height[i] is the new border, the area of ith bin should be 0
                  example:
                  [2, 10, 5], for 1th bin, if we do not inlcude 10, the area would be negative.
                  """
                  left_max[i] = max(left_max[i-1], height[i])

              for i in range(n-2, -1, -1):
                  right_max[i] = max(right_max[i+1], height[i])

              area = 0
              for i in range(1, n-1):  # from 1 to n-2
                area += min(left_max[i], right_max[i]) - height[i]

              return area
            ```
      * Approach2: greedy with 2 pointers, Time: O(n), Space: O(1)
          * Concept
            * If the right border > left border, the area of ith bin is determined by the left border.
            * vice versa
            * So we fix the higher border, and move the lower border to calcualte the area
          * Python Solution
            ```python
            def trap(self, height: List[int]) -> int:
              if not height:
                  return 0

              area = 0
              left, right = 0, len(height)-1
              max_left = max_right = 0
              while left < right:
                  # area depends on left border
                  if height[left] <= height[right]:
                      if height[left] >= max_left:
                          # new left border, do not need to calculate area
                          max_left = height[left]
                      else:
                          area += (max_left - height[left])
                      left += 1

                  # area depends on right border
                  else:
                      if height[right] > max_right:
                          # new right border, do not need to calculate area
                          max_right = height[right]
                      else:
                          area += (max_right - height[right])
                      right -= 1

              return area
            ```
  * **Jump Game**:
    * 055: Jump Game (M)
      * Ref:
        * https://leetcode.com/problems/jump-game/solution/
      * Approach1: DP, Recursive + memo (top-down), Time: O(n^2), Space: O(n)
        * Python Solution
          ```python
          def canJump(self, nums: List[int]) -> bool:
            def _can_jump_from(start):
                # can not use True/False evaluation
                if memo[start] is not None:
                    return memo[start]

                memo[start] = False
                max_jump_dst = min(start+nums[start], n-1)

                # from max_jump_dst to start + 1
                for jump_dst in range(max_jump_dst, start, -1):
                    if _can_jump_from(jump_dst):
                        memo[start] = True
                        break

                return memo[start]

            if not nums:
                return False

            n = len(nums)
            memo = [None] * n
            memo[n-1] = True

            return _can_jump_from(0)
          ```
      * Approach2: DP, Iterative + memo (botoom-up), Time: O(n^2), Space: O(n)
        * Python Solution
          ```python
          def canJump(self, nums: List[int]) -> bool:
            n = len(nums)
            memo = [None] * n
            memo[n-1] = True

            # from n-2 to 0
            for start in range(n-2, -1, -1):
                max_jump_dst = min(start+nums[start] ,n-1)
                # from max_jump_dst to start+1
                for jump_dst in range(max_jump_dst, start, -1):
                    if memo[jump_dst]:
                        memo[start] = True
                        break

            return memo[0]
          ```
      * Approach2: Greedy, Time: O(n), Space: O(1)
        * **The main concept is to keep the left most good index**
          * If we can reach a GOOD index, then our position is a GOOD index as well. and this new GOOD index will be the new leftmost GOOD index.
        * Python Solution
          ```python
          def canJump(self, nums: List[int]) -> bool:
            n = len(nums)
            left_most_good_idx = n - 1

            # from n-2 to 0
            for start in range(n-2, -1, -1):
                if start + nums[start] >= left_most_good_idx:
                    left_most_good_idx = start

            return left_most_good_idx == 0
          ```
    * 045: Jump Game II (H) *
      * Ref:
        * https://leetcode.com/problems/jump-game-ii/discuss/18014/Concise-O(n)-one-loop-JAVA-solution-based-on-Greedy
      * Approach1: Greedy
        * Find the minimum jump
        * Greedy
          * Time: O(n), Space: O(1)
          *  **cur == cur_border**, like BFS solution
             *  means you visited all the items on the current level
             *  Incrementing jumps+=1 is like incrementing the level you are on.
          *  And **cur_end = cur_farthest** is like getting the queue size (level size) for the next level you are traversing.
        *  Python Solution
            ```python
            def jump(self, nums: List[int]) -> int:
              jump_cnt = cur_border = cur_farthest = 0
              # from 0 to n-2
              for cur in range(0, len(nums)-1):
                  cur_farthest = max(cur_farthest, cur+nums[cur])
                  # the boundary need to jump
                  if cur == cur_border:
                      jump_cnt +=1
                      # determine the next border
                      cur_end = cur_farthest

              return jump_cnt
            ```
  * **H-Index**
    * 274: H-Index (M)
      * observation:
        * max_hidx = len(nums)
        * possible h_idx is between 0 ~ max_hidx
      * Approach1: Use Array to memo, Time O(n), Space O(n)
        * Concept
          * **The max index in the array would be len(array)**, that is we can restrict the number of the buckets.
        * Use array to keep the cnt of citations
        * Python
            ```python
            def hIndex(self, citations: List[int]) -> int:
              max_hidx = len(citations)
              memo = [0] * (max_hidx + 1)
              for cita in citations:
                  memo[min(cita, max_hidx)] += 1

              res = 0
              cita_cnt = 0
              for h_idx in range(max_hidx, -1, -1):
                  cita_cnt += memo[h_idx]
                  if cita_cnt >= h_idx:
                      res = h_idx
                      break

              return res
            ```
      * Approach2: Use Sort, Time: O(nlog(n)), Space: O(sorting)
        * Concept:
          * step1:
            * Sorting costs O(nlog(n))
          * step2:
            * Find the H-Index costs O(log(n))
            * please refer 275: H-Index II
    * H-Index II (M)
      * Please refer binary search
  * **Best Time to Buy and Sell Stock**
    * Ref:
      * [General solution](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/discuss/108870/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems)
    * 121: Best Time to Buy and Sell Stock (E)
      * You may complete **at most one transactions**.
      * Approach1: Time:O(n), Space:O(1)
        * For each round, keep the current minimum buy price and update best sell prcie.
        * Python Solution:
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            if not prices:
                return  0

            max_profit = 0
            min_price = prices[0]

            for i in range(1, len(prices)):
                profit = prices[i] - min_price
                max_profit = max(profit, max_profit)
                min_price = min(prices[i], min_price)

            return max_profit
          ```
    * 122: Best Time to Buy and Sell Stock II (E)
      * Multiple transcations allowed.
      * Approach1: Peak Valley , Time:O(n), Space:O(1)
        * Python Solution:
        ```python
        def maxProfit(self, prices: List[int]) -> int:
          """
          Peak Valley Approach
          """
          n = len(prices)
          valley = 0
          profit = 0

          while valley < n:
              while valley + 1 < n and prices[valley+1] <= prices[valley]:
                  valley += 1

              peak = valley + 1
              while peak + 1 < n and prices[peak+1] >= prices[peak]:
                  peak += 1

              if valley < peak < n:
                  profit += (prices[peak]-prices[valley])

              valley = peak + 1

          return profit
        ```
      * Approach2: Peak Valley II Time:O(n), Space:O(1)
        * Python Solution1
           ```python
           valley = peak = prices[0];
           max_profit = 0
           while i < len(prices) - 1:
              # find the valley
              while i < len(prices) - 1 and price[i] >= prices[i+1]:
                  i += 1
              valley = prices[i]

              while i < len(prices) - 1 and price[i] <= prices[i+1]:
                  i += 1
              peak = prices[i]

              max_profit += (peak - valley);
           ```
        * Python Solution2
          ```python
          max_profit = 0
          for i in range(1, len(prices)):
              if prices[i] > price[i-1]:
                  max_profit += prices[i] - price[i-1]
          ```
    * 714: Best Time to Buy and Sell **Stock with Transaction Fee** (M), Time:O(n), Space:O(1)
      * Ref:
        * [solution](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/solution/)
      * Approach1: bottom up DP, Time:O(n), Space:O(1)
        * Cash(i):
          * The cash in hand, if you are **not holding the stock** at the end of day(i):
            * case1:
              * cash[i] = cash[i-1]
            * case2:
              * cash[i] = hold[i-1] + prcie[i] - fee
            * cash[i]
              * max(case1, case2) = max(cash[i-1], hold[i-1] + prcie[i] - fee)
        * Hold(i):
          * The cash in hand, if you are **holding the stock** at the end of day(i):
            * case1:
              * hold[i] = hold[i-1]
            * case2:
              * hold[i] = **hold[i-1] + price[i] - fee** - price[i]
            * case3:
              * hold[i] = **cash[i-1]** - price[i]
            * case2 and case3 can be reduced to
              *  **cash[i]** - price[i]
            * hold[i]
              * max(case1, case2, case3) = max(hold[i-1], **cash[i]-price[i]**)
        * Python Solution
            ```python
            def max_profit(self, prices: List[int], fee: int) -> int:
              cash = 0
              hold = -prices[0]
              for i in range(1, len(prices)):
                  """
                  cash[i] = max(case1, case2)
                  case1: cash[i-1]
                  case2:  hold[i-1] + prices[i] - fee
                  """
                  cash = max(cash, hold+prices[i]-fee)
                  """
                  hold[i] = max(case1, case4)
                  case1: hold[i-1]
                  case2: case[i-1] - prices[i-1]
                  case3: hold[i-1] + prices[i-1] - fee -prices[i-1]
                  case4: cash[i] - prices[i-1]  (come from case2 and case3)
                  """
                  hold = max(hold, cash-prices[i])
              return cash
            ```
    * 123: Best Time to Buy and Sell Stock III (H)
      * You may complete **at most two transactions**.
    * 188: Best Time to Buy and Sell Stock IV (H)
      * You may complete **at most k transactions**.
  * **Shortest Word Distance**
    * 243: Shortest Word Distance (E) *
       * Approach1: Time:O(n), Space:O(1)
         * Calculate the distance and update the shortest distance in each round.
           * Draw some cases1:
             * 1 2 1* -> will cover two cases 1-2 and 2-1*
             * 1 2 2* 1* -> still can cover
               * 1-2* will not cover, but it is bigger than 1-2
               * 2-1* is the same, it is bigger than 2*-1*
           * Python Solution
              ```python
              def shortestDistance(self, words: List[str], word1: str, word2: str) -> int:
                shortest_dist = len(words) # beyond upper bound
                idx1 = idx2 = -1
                found = False

                for idx, w in enumerate(words):
                    if w == word1:
                        idx1 = idx

                    elif w == word2:
                        idx2 = idx

                    if idx1 == -1 or idx2 == -1:
                        continue

                    shortest_dist = min(shortest_dist, abs(idx1-idx2))
                    found = True

                return shortest_dist if found else -1
              ```
    * 245: Shortest Word Distance III (M) *
      * Allow **duplicated words**.
      * Approach1:
        * The same concept with 243, but need to handle duplicated words
        * Python Solution
            ```python
            def shortestWordDistance(self, words: List[str], word1: str, word2: str) -> int:
              shortest_dist = len(words)
              idx1 = idx2 = -1
              found = False
              same = True if word1 == word2 else False

              for idx, w in enumerate(words):
                  if w == word1:
                      if same:
                          idx1, idx2 = idx2, idx
                      else:
                          idx1 = idx
                  elif w == word2:
                      idx2 = idx

                  if idx1 == -1 or idx2 == -1:
                      continue

                  shortest_dist = min(shortest_dist, abs(idx1-idx2))
                  found = True

              return shortest_dist if found else -1
            ```
    * 244: Shortest Word Distance II (M) **
       * **Init once** and **search for multiple time**.
       * Approach1: Time:O(K + L), Space:O(n)
         * Using **Preprocessed Sorted Indices** and two pointers to traverse
           * Space: O(n)
             * For or the dictionary that we prepare in the constructor.
             * The keys represent all the unique words in the input and the values represent all of the indices from 0 ... N0...N.
         * Time Complexity:
           * Init step:
             * O(n), where n is the number of words.
           * Find the shortest distance :
             * O(K + L), where K and L represent the number of occurrences of the two words.
         * Python Solution
            ```python
            class WordDistance:
              def __init__(self, words: List[str]):
                  self.word_d = collections.defaultdict(list)
                  for idx, w in enumerate(words):
                      self.word_d[w].append(idx)

              def shortest(self, word1: str, word2: str) -> int:
                  idx_list_1 = self.word_d[word1]
                  idx_list_2 = self.word_d[word2]
                  idx1 = idx2 = 0
                  shortest_dist = float('inf')

                  while idx1 < len(idx_list_1) and idx2 < len(idx_list_2):
                      w_idx1 = idx_list_1[idx1]
                      w_idx2 = idx_list_2[idx2]
                      shortest_dist = min(shortest_dist, abs(w_idx1-w_idx2))
                      # move the smaller one
                      if w_idx1 <= w_idx2:
                          idx1 += 1
                      else:
                          idx2 += 1

                  return shortest_dist
            ```
  * **Interval**
    * 252: Meeting Rooms (E)
      * Check if one person **can attend all meetings**.
      * How to check overlaps ?
        * min(interval1.end, interval2.end) > max(interval1.start, interval2.start)
      * Approach1: Brute Force, Time:O(n^2)
          * Python Solution
            ```python
            def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
              start, end = 0, 1
              can_attend = True

              # (0, 1), (0, 2), (0, 3)
              #         (1, 2), (1, 3)
              #                 (2, 3)
              for i in range(0, len(intervals)-1):
                  for j in range(i+1, len(intervals)):
                      if min(intervals[i][end], intervals[j][end]) > max(intervals[i][start], intervals[j][start]):
                          can_attend = False
                          break

              return can_attend
            ```
      * Approach2: Check after sorting, Time:O(nlog(n))
        * For **sorted intervals**
          * Check the overlap between interval[i] and interval[i+1] would be
            * If there is an overlap between interval 3 and interval1, then there is an overlap between interval2 and interval1 as well, since interval2.start < interval3.start.
        * Algo:
          * Sort by start time of intervals
          * Check if interval[i] and intervalp[i+1] have overlap.
        * Python Solution
          ```python
          def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
            start, end = 0, 1
            can_attend = True
            intervals.sort(key=lambda interval: interval[start])

            for i in range(1, len(intervals)):
                if intervals[i-1][end] > intervals[i][start]:
                    can_attend = False

            return can_attend
          ```
    * 253: Meeting Rooms II (M)
      * Find the minimum requirement of the meeting rooms.
      * Approach1: Brute Force, Time: O(n^2), Space: O(1)
      * Approach2: Min Heap: O(nlog(n)), Space: O(n)
        * Check after sorting
        * Algo
          * Sort the intervals by start time
          * For every meeting room check if the minimum element of the heap is free or not.
            * If the room is free, then we extract the topmost element and add it back with the ending time of the current meeting we are processing.
            * If not, then we allocate a new room and add it to the heap.
        * Python Solution
            ```python
            def minMeetingRooms(self, intervals: List[List[int]]) -> int:
              if not intervals:
                  return 0

              start, end = 0, 1
              intervals.sort(key=lambda interval: interval[start])
              heap = [intervals[0][end]]

              for i in range(1, len(intervals)):
                  # heap[0] is current minimum end time
                  # need a new room
                  if heap[0] > intervals[i][start]:
                      heapq.heappush(heap, intervals[i][end])
                  # reuse the room
                  else:
                      heapq.heapreplace(heap, intervals[i][end])

              return len(heap)
            ```
    * 056: Merge Intervals (M)
      * Approach1: Sorting and check, Time: O(nlogn), Space: O(n)
        * Sort the intervals by start time,
        * Update the output if there exists interval overlap.
        * Python Solution
          ```python
          def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            if not intervals:
                return []

            start, end = 0, 1
            intervals.sort(key=lambda interval: interval[start])
            output = [intervals[0][:]]

            for i in range(1, len(intervals)):
                last = output[-1]
                cur = intervals[i]

                # merge
                if last[end] >= cur[start]:
                      last[end] = max(last[end], cur[end])
                else:
                    # copy
                    output.append(cur[:])

            return output
          ```
    * 057: Insert Interval (H)
    * 352: Data Stream as Disjoint Intervals (H)
  * **Subarray**
    * 053: Maximum Subarray (E)
       * Approach1: Brute Force, O(n^2)
         * Python Solution
            ```python
            def maxSubArray(self, nums: List[int]) -> int:
              max_sum = 0
              for start in range(len(nums)):
                  max_sum_so_far = 0
                  for end in range(start, len(nums)):
                      max_sum_so_far += nums[end]
                      max_sum = max(max_sum_so_far, max_sum)

              return max_sum
            ```
       * Approach2: Kadane's Algorithm, O(n)
         * Python Solution
            ```python
            def max_sub_array(self, nums: List[int]) -> int:
                max_sum = max_sum_so_far = nums[0]

                for i in range(1, len(nums)):
                    max_sum_so_far = max(nums[i], max_sum_so_far+nums[i])
                    max_sum = max(max_sum, max_sum_so_far)

                return max_sum
          ```
    * 152: Maximum **Product** Subarray (M)
      * Approach1: Brute Force, Time:O(n^2)
      * Approach2: Kadane's Algorithm like algo, Time: O(n), Space: O(1)
        * The concept is just like 53: maximum subarray, but need to notice negative value in the array.
        * Python Solution
          ```python
          def maxProduct(self, nums: List[int]) -> int:
            g_max = cur_min = cur_max = nums[0]
            for i in range(1, len(nums)):
                # we redefine the extremums by swapping them
                # when multiplied by a negative
                if nums[i] < 0:
                    cur_min, cur_max = cur_max, cur_min

                cur_max = max(nums[i], cur_max*nums[i])
                cur_min = min(nums[i], cur_min*nums[i])

                g_max = max(cur_max, g_max)

            return g_max
          ```
    * 325: Maximum Size Subarray Sum **Equals k** (M)
      * Find the maximum length of a subarray that sums to k
      * Approach1: Brute force, O(n^2), Space: O(1)
        * List all pairs of acc and keep the max
        * Python Solution
          ```python
          def subarraySum(self, nums: List[int], k: int) -> int:
            max_len = 0
            for start in range(0, len(nums)):
              acc = 0 # reuse the acc
              for end in range(start, len(nums)):
                acc += nums[s]
                if acc == k:
                  max_len = max(end-start, max_len)
            return max_len
          ```
      * Approach2: Hash Table, Time: O(n), Space: O(n)
        * [Concept](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/discuss/77784/O(n)-super-clean-9-line-Java-solution-with-HashMap)
          * Use hash table
            * key: accumulation value
            * val: index
        * Use hash table to keep acc value and the index.
        * Python Solution
          ```python
          def maxSubArrayLen(self, nums: List[int], k: int) -> int:
            acc = max_len = 0
            # key:   accumulate value
            # value: index of the accumulate value
            d = {0: -1}

            for idx, n in enumerate(nums):
                acc += n
                target_acc = acc - k  # target_acc + k = acc
                if target_acc in d:
                    max_len = max(max_len, idx-d[target_acc])

                # since we need the max diff, only update when there is not existing.
                if acc not in d:
                    d[acc] = idx

            return max_len
          ```
    * 560: Subarray Sum **Equals K**, (M) ***
      * Approach1: brute force, O(n^2), Space: O(1)
        * List all pairs of acc and keep the max
        * python solution
          ```python
          def subarraySum(self, nums: List[int], k: int) -> int:
            cnt = 0
            for start in range(0, len(nums)):
              acc = 0 # reuse the acc
              for end in range(start, len(nums)):
                acc += nums[s]
                if acc == k:
                  cnt += 1
            return cnt
          ```
      * Approach2: Use hash Table Time: O(n), Space: O(n)
        * Use hash table to keep acc value and cnt
        * Python Solution
          ```python
          def subarraySum(self, nums: List[int], k: int) -> int:
            acc = cnt = 0
            d = collections.defaultdict(int)
            d[0] += 1 # init value for acc == k

            for n in nums:
              acc += n
              target_acc = acc - k

              if target_acc in d:
                  cnt += d[target_acc]

              d[acc] += 1
            return cnt
          ```
    * 209: Minimum Size Subarray Sum (M)
      * Approach1: Brute Force, Time:O(n^2), Space:O(1)
        * Python
          ```python
          def minSubArrayLen(self, s: int, nums: List[int]) -> int:
            if not nums:
              return 0

            min_len = float('inf')
            for start in range(len(nums)):
                acc = 0
                for i in range(start, len(nums)):
                    acc += nums[i]
                    if acc >= s:
                        min_len = min(min_len, i-start+1)
                        break

            if min_len == float('inf'):
              min_len = 0

            return min_len
          ```
      * Approach2: Binary Search, Time:O(nlogn), Space:O(n)
        * binary search:
          * 1. create a memo, memo[i] = memo[i-1] + nums[i]
          * 2. for i in range(0, n):
            * find the minimum j where:
              * memo[j] - memo[i] >= target
              * memo[j] >= memo[i] + target
                * mid >= memo[i] + target
                  * r = mid - 1
                * mid < memo[i] + target
                  * l = mid + 1
                * return left
      * Approach3: Sliding Window, Time:O(n), Space:O(1)
        * Python
          ```python
          def minSubArrayLen(self, s: int, nums: List[int]) -> int:
            if not nums:
                return 0

            min_len = float('inf')
            start = end = 0
            acc = 0

            while end < len(nums):
                acc += nums[end]
                while start <= end and acc >= s:
                    min_len = min(min_len, end-start+1)
                    acc -= nums[start]
                    start += 1

                end += 1

            if min_len == float('inf'):
                min_len = 0

            return min_len
          ```
    * 238: **Product** of Array **Except Self** (M)
      * Approach1: Allow to use Division, Time:O(n)
        * We can simply take the product of all the elements in the given array and then, for each of the elements xx of the array, we can simply find product of array except self value by dividing the product by xx.
        * Containing zero cases
          * exactly 1 zero
          * more than 1 zero
        * Python Solution,
          ```python
          def productExceptSelf(self, nums: List[int]) -> List[int]:
            product = 1
            zero_idx = zero_cnt = 0
            output = [0] * len(nums)

            for idx, n in enumerate(nums):
                if n == 0:
                    zero_cnt += 1
                    zero_idx = idx
                else:
                    product *= n

            if zero_cnt == 1:
              output[zero_idx] = product
            elif zero_cnt == 0:
              for i in range(len(output)):
                output[i] = product//nums[i]
            return output
            ```
      * Approach2: Not Allow to use Division1: Time:O(n), Space:O(n)
        * **For every given index i, we will make use of the product of all the numbers to the left of it and multiply it by the product of all the numbers to the right**.
        * Python Solution:
        ```python
        def productExceptSelf(self, nums: List[int]) -> List[int]:
          n = len(nums)
          output = [0] * n
          left_p = [1] * n
          right_p = [1] * n

          # 1 to n-1
          for i in range(1, n):
              left_p[i] = left_p[i-1] * nums[i-1]

          # n-2 to 0
          for i in range(n-2, -1, -1):
              right_p[i] = right_p[i+1] * nums[i+1]

          for i in range(0, n):
              output[i] = left_p[i] * right_p[i]

          return output
        ```
      * Approach3: Not Allow to use Division2: Time:O(n), Space:O(n)
          * We can use a variable to replace right_product array
          * Python Solution:
            ```python
            def productExceptSelf(self, nums: List[int]) -> List[int]:
              n = len(nums)
              output = [0] * n
              left_p = [1] * n

              # 1 to n-1
              for i in range(1, n):
                  left_p[i] = left_p[i-1] * nums[i-1]

              right = 1
              for i in range(n-1, -1, -1):
                  output[i] = left_p[i] * right
                  right *= nums[i]

              return output
            ```
    * 228: **Summary Ranges** (M)
      * Given a sorted integer array without duplicates, return the summary of its ranges.
        * Input:
          * [0,1,2,4,5,7]
        * Output:
          * ["0->2","4->5","7"]
      * Approach1:
        * Python Solution
          ```python
          def summaryRanges(self, nums: List[int]) -> List[str]:
            n = len(nums)
            output = list()
            start = 0

            while start < n:
                end = start
                while end + 1 < n and nums[end]+1 == nums[end+1]:
                    end += 1

                if end == start:
                    output.append(str(nums[start]))
                else:
                    output.append(f'{nums[start]}->{nums[end]}')

                start = end + 1

            return output
          ```
    * 163: **Missing Ranges** (M)
      * Example:
        * Input: nums = [0, 1, 3, 50, 75], lower = 0 and upper = 99
        * Output: ["2", "4->49", "51->74", "76->99"]
      * Approach1:
        * Need to know how to handle the boundary properly.
        * Python Solution
        ```python
          @staticmethod
          def format_result(left_boundary, right_boundary, output):
            diff = right_boundary - left_boundary
            if diff == 2:
                output.append(str(left_boundary+1))
            elif diff > 2:
                output.append(f"{left_boundary+1}->{right_boundary-1}")

          def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
              output = []
              left_boundary = lower - 1
              for n in nums:
                  right_boundary = n
                  self.format_result(left_boundary, right_boundary, output)
                  left_boundary = right_boundary

              right_boundary = upper + 1
              self.format_result(left_boundary, right_boundary, output)
              return output
          ```
    * 239: Sliding Window Maximum (H)
  * **Reorder**
    * 189: Rotate Array (E)
      * Approach1: Time:O(n), Space:O(1)
        * Use **three reverse** operations can solve this problem.
        * Python Solution
          ```python
          def reverse(nums: List[int], start, end) -> None:
            while start < end:
              nums[start], nums[end] = nums[end], nums[start]
              start +=1
              end -=1

          def rotate(self, nums: List[int], k: int) -> None:
            """
            Do not return anything, modify nums in-place instead.
            """
            if not nums or not k:
                return

            n = len(nums)
            k = k % n
            if k == 0:
                return

            reverse(nums, 0, n-1)
            reverse(nums, 0, k-1)
            reverse(nums, k, n-1)
          ```
    * 088: Merge Sorted Array (E)
      * You may **assume that nums1 has enough space** (size that is greater or equal to m + n) to hold additional elements from nums2.
      * Approach1: Space O(1):
        * Fill the arrary **from the end to the start**
        * Python Solution:
        ```python
        def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
          """
          Do not return anything, modify nums1 in-place instead.
          """
          merge_runner = len(nums1) - 1
          runner1 = m - 1
          runner2 = n - 1

          while runner1 >= 0 and runner2 >= 0:
              if nums1[runner1] >= nums2[runner2]:
                  nums1[merge_runner] = nums1[runner1]
                  runner1 -= 1
              else:
                  nums1[merge_runner] = nums2[runner2]
                  runner2 -= 1
              merge_runner -= 1

          while runner2 >= 0:
              nums1[merge_runner] = nums2[runner2]
              runner2 -= 1
              merge_runner -=1
        ```
    * 283: Move Zeroes (E)
      * **move all 0's to the end** of it while maintaining the relative order of the non-zero elements.
      * Approach1
        * Like the partition step of quick sort
          * **keep the border pointing to next available position.**
          * Python Solution
          ```python
          def moveZeroes(self, nums: List[int]) -> None:
            """
            Do not return anything, modify nums in-place instead.
            """
            if len(nums) <= 1:
                return

            border = 0
            for i in range(0, len(nums)):
                if nums[i] == 0:
                    continue
                nums[border], nums[i] = nums[i], nums[border]
                border += 1
           ```
  * Intersection:
    * 349: Intersection of Two Arrays (E)
      * Each element in the result must be unique.
      * Approach1: Use hash Table, Time:O(m+n), Space:O(m)
        * Python
          ```python
          def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
            def get_d(nums):
                d = dict()
                for n in nums:
                    d[n] = True
                return d

            d1 = get_d(nums1)

            intersection = []
            for n in nums2:
                if n in d1 and d1[n]:
                    intersection.append(n)
                    d1[n] = False

            return intersection
          ```
    * 350: Intersection of Two Arrays II (E)
      * Each element in the result should appear as many times as it shows in both arrays.
      * follow up:
        * What if the given array is already sorted? How would you optimize your algorithm?
      * Approach1: Hash Table, Time:O(m+n), Space:O(m)
        * Python
          ```python
          def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
            d1 = collections.Counter(nums1)
            intersection = []
            for n in nums2:
                if n in d1 and d1[n] > 0:
                    intersection.append(n)
                    d1[n] -= 1

            return intersection
          ```
      * Approach2, if the array is already sorted: Time:O(m+n), Space:O(1)
          * Python
            ```python
            def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
              nums1.sort()
              nums2.sort()
              p1 = p2 = 0
              intersection = []

              while p1 < len(nums1) and p2 < len(nums2):
                  v1, v2 = nums1[p1], nums2[p2]
                  if v1 == v2:
                      p1 += 1
                      p2 += 1
                      intersection.append(v1)

                  elif v1 < v2:
                      p1 += 1

                  else:
                      p2 += 1

              return intersection
            ```
  * Other:
    * 277: Find the Celebrity (M)
      * Ref:
        * https://pandaforme.github.io/2016/12/09/Celebrity-Problem/
      * Approach1:
        1. Find the **celebrity candidate**
        2. Check if the candidate is the celebrity
           * Check the people before the celebrity candidate:
              * The celebrity does not know them but they know the celebrity.
           * Check the people after the celebrity candidate:
             * They should know the celebrity
         * Python Solution
            ````python
            # Return True if a knows b
            def knows(a,  b):
              pass

            def find_celebrity(self, n):
                """
                :type n: int
                :rtype: int
                """
                unknown = -1
                celebrity = 0
                # find the celebrity candidate
                for p in range(1, n):
                    if not knows(celebrity, p):
                        continue
                    celebrity = p

                # check people in the left side
                for p in range(celebrity):
                    if knows(p, celebrity) and not knows(celebrity, p):
                        continue
                    return unknown

                # # check people in the right side
                for p in range(celebrity+1, n):
                    if knows(p, celebrity):
                        continue
                    return unknown

                return celebrity
            ````
    * 041: First missing positive (H)
      * Ref
        * https://leetcode.com/problems/first-missing-positive/discuss/17073/Share-my-O(n)-time-O(1)-space-solution
        * https://leetcode.com/problems/first-missing-positive/discuss/17071/My-short-c%2B%2B-solution-O(1)-space-and-O(n)-time
      * The idea is **like you put k balls into k+1 bins**, there must be a bin empty, the empty bin can be viewed as the missing number.
      * For example:
        * if length of n is 3:
          * The ideal case would be: [1, 2, 3], then the first missing is 3+1 = 4
          * The missing case may be: [1, None, 3]  5, then the first missing is 2
      * Approach1: Time O(n), Space O(n)
        * Use extra space to keep the sorted positve numbers.
        * Python Solution
          ```python
          def firstMissingPositive(self, nums: List[int]) -> int:
            n = len(nums)
            sorted_nums = [None] * n

            for num in nums:
                if 0 < num <= n:
                    sorted_nums[num-1] = n

            res = n + 1 # not found
            for i in range(0, n):
                if not sorted_nums[i]:
                    res = i+1
                    break

            return res
          ```
      * Approach2: Time O(n), Space O(1)
         1. Each number will be put in its right place at most once after first loop *
         2. Traverse the array to find the unmatch number
         * Python Solution
            ```python
            def firstMissingPositive(self, nums: List[int]) -> int:
              n = len(nums)

              for i in range(n):
                  # We visit each number once, and each number will
                  # be put in its right place at most once
                  while 0 < nums[i] <= n and nums[nums[i]-1] != nums[i]:
                      # the correct position of nums[i] is in
                      # nums[nums[i]#-1]
                      correct = nums[i]-1  # need this correct
                      nums[i], nums[correct] = nums[correct] , nums[i]

              res = n + 1
              for i in range(n):
                  if n[i] != i+1:
                      res i+1
                      break
              # not found from 0 to length-1, so the first missing is in length-th
              return res
            ```
    * 299: Bulls and Cows (M)
      * Bull:
        * Two characters are the same and having the same index.
      * Cow
        * Two characters are the same but do not have the same index.
      * Approach1: Hash TableTime O(n), Space O(n) and **one pass**
        * Use **hash Table** to count cows.
        * Python solution
          ```python
          def getHint(self, secret: str, guess: str) -> str:
            bull, cow = 0, 0
            d = collections.defaultdict(int)

            for s, g in zip(secret, guess):

                if s == g:
                    bull += 1
                else:
                    if d[s] < 0:
                        cow += 1

                    if d[g] > 0:
                        cow += 1

                    d[s] += 1
                    d[g] -= 1

            return f'{bull}A{cow}B'
          ```
    * 134: Gas Station (M)
        * Time: O(n)
          * Concept:
            * rule1:
              * **If car starts at A and can not reach B. Any station between A and B can not reach B.**
              * If A can't reach B, and there exists C between A & B which can reach B, then A can reach C first, then reach B from C, which is conflict with our init statement: A can't reach B. so, the assume that such C exists is invalid.
            * rule2
              * **If the total number of gas is bigger than the total number of cost, there must be a solution.**
              * [Proof](https://leetcode.com/problems/gas-station/discuss/287303/Proof%3A-if-the-sum-of-gas-greater-sum-of-cost-there-will-always-be-a-solution)
          * Python Solution
            ```python
            not_found = -1
            start = 0
            total_tank = 0
            cur_tank = 0

            for i in range(len(gas)):
                remain = gas[i] - cost[i]
                cur_tank += remain
                total_tank += remain

                # try another start
                # rule2
                if cur_tank < 0:
                    # rule1
                    start = i+1
                    cur_tank = 0

            return start if total_tank >= 0 else not_found
            ```
    * 289: Game of Life (M)
      * Approach1: Time: O(mn), Space: O(mn)
        * Python Solution
          ```python
          def gameOfLife(self, board: List[List[int]]) -> None:
            # coordinate diff for 8 neighbors
            neighbors = [(1, 0), (1, -1), (0, -1), (-1, -1),
                        (-1, 0), (-1,1), (0, 1), (1, 1)]

            rows = len(board)
            cols = len(board[0])
            copy_board = [[board[row][col] for col in range(cols)] for row in range(rows)]

            for row in range(rows):
                for col in range(cols):
                    # calculate the cnt of live neighbors
                    live_neighbors = 0
                    for n in neighbors:
                        r, c = row + n[0], col + n[1]
                        if 0 <= r < rows and 0 <= c < cols \
                          and copy_board[r][c] == 1:
                            live_neighbors += 1

                    # change status
                    if copy_board[row][col] == 1:
                        if live_neighbors < 2 or live_neighbors > 3:
                            board[row][col] = 0

                    else: # ref_board[row][col] == 0
                        if live_neighbors == 3:
                            board[row][col] = 1
          ```
      * Approach2: Time: O(mn), Space: O(1)
        * Use two temp status, live_2_dead and dead_2_live
        * Python Solution
          ```python
          class Status(object):
            live_2_dead = -1
            dead = 0
            live = 1
            dead_2_live = 2

          def gameOfLife(self, board: List[List[int]]) -> None:
            # coordinate diff for 8 neighbors
            neighbors = [(1, 0), (1, -1), (0, -1), (-1, -1),
                        (-1, 0), (-1,1), (0, 1), (1, 1)]

            rows = len(board)
            cols = len(board[0])

            for row in range(rows):
                for col in range(cols):
                    # calculate the cnt of live neighbors
                    live_neighbors = 0
                    for n in neighbors:
                        r, c = row + n[0], col + n[1]
                        if 0 <= r < rows and 0 <= c < cols \
                          and abs(board[r][c]) == 1:  # Status.live and Status.live_2_dead
                            live_neighbors += 1

                    # change status
                    if board[row][col] == Status.live:
                        if live_neighbors < 2 or live_neighbors > 3:
                            board[row][col] = Status.live_2_dead

                    else: # ref_board[row][col] == 0
                        if live_neighbors == 3:
                            board[row][col] = Status.dead_2_live

            for row in range(rows):
                for col in range(cols):
                    if board[row][col] > 0: # live and live to dead
                        board[row][col] = Status.live
                    else:                   # dead and dead to live
                        board[row][col] = Status.dead
          ```
      * follow up: Infinite array
        * [Solution](https://leetcode.com/problems/game-of-life/discuss/73217/Infinite-board-solution/201780)
    * 723：Candy Crush (M)
### Matrix
 * **Rotate**:
   * Approach1: m*n arryay, Time:O(mn), Space:O(mn)
     * Python
      ```python
      def rotate_v2(a):
        """
        non-np version

        Complexity:
        Time  : O(rc)
        Space : O(rc)
        """
        row_num = len(a)
        col_num = len(a[0])

        rotated_array = [[None for _ in range(row_num)] for _ in range(col_num)]

        for r in range(row_num):
            for c in range(col_num):
                rotated_array[c][row_num-1-r] = a[r][c]

        return rotated_array
      ```
   * Approach2: n-n array, Time:O(mn), Space:O(1)
     * Python
      ```python
      def rotate_in_place(a):
        row_num = len(a)
        col_num = len(a[0])

        if row_num != col_num:
            raise ArgError()

        dim = row_num

        layers = row_num // 2

        for layer in range(layers):
            start = layer
            end = dim - 1 - layer

            for offset in range(end-start):
                tmp = a[start][start+offset]
                a[start][start+offset] = a[end-offset][start]
                a[end-offset][start] = a[end][end-offset]
                a[end][end-offset] = a[start+offset][end]
                a[start+offset][end] = tmp

        return a
      ```
 * **Spiral**
   * 054:	Spiral Matrix (M)
     * Approach1: Simulation, Time:O(mn), Space:O(mn)
       * Draw the path that the spiral makes. We know that the path should turn clockwise whenever it would go out of bounds or into a cell that was previously visited.
       * Python
        ```python
        def spiralOrder(self, matrix):
          if not matrix:
              return []

          row, col = len(matrix), len(matrix[0])
          seen = [[False for _ in range(col)] for _ in range(row)]

          res = []
          r = c = d = 0
          # top, right, bottom, left
          directions = ((0, 1), (1, 0), (0,-1), (-1, 0))

          for _ in range(row*col):
              res.append(matrix[r][c])
              seen[r][c] = True
              cr, cc = r + directions[d][0], c + directions[d][1]

              if 0 <= cr < row and 0 <= cc < col and not seen[cr][cc]:
                  r, c = cr, cc
              else:
                  # change direction when out of bounds or into a cell that was
                  # previously visited
                  d = (d + 1) % 4
                  r, c = r + directions[d][0], c + directions[d][1]

          return res
        ```
     * Approach2: Layer by Layer, Time:O(mn), Space:O(1)
       * Ref:
         * https://leetcode.com/articles/spiral-matrix/
       * For each layer
         * 1. Top
         * 2. Right
         * 3. Bottom
         * 4. Left
         * Notes: need to handle single row, single col and one unit case
       * Python
        ```python
        def spiralOrder(self, matrix):
          def spiral_coords(r1, c1, r2, c2):
              # Top, go right, c1 to c2
              for c in range(c1, c2+1):
                  yield r1, c

              # Right, go down , r1+1 to r2
              for r in range(r1 + 1, r2 + 1):
                  yield r, c2

              # prevent duplicate for single row and single col
              if r1 < r2 and c1 < c2:
                  # Bottom, go left, c2-1 to c1
                  for c in range(c2-1, c1-1, -1):
                      yield r2, c

                  # Left, go up  r2-1 to r1-1
                  for r in range(r2-1, r1, -1):
                      yield r, c1

          if not matrix:
              return []

          row, col = len(matrix), len(matrix[0])
          # top left corner
          r1 = c1 = 0
          # bottom right corner
          r2, c2 = row-1, col-1
          res = []

          while r1 <= r2 and c1 <= c2:
              for r, c in spiral_coords(r1, c1, r2, c2):
                  res.append(matrix[r][c])

              # move to next layer
              r1, c1 = r1 + 1, c1 + 1
              r2, c2 = r2 - 1, c2 - 1

          return res
        ```
   * 059:	Spiral Matrix II (M)
     * Approach1:Simulation, Time:O(mn), Space:O(1)
       * Python
        ```python
        def generateMatrix(self, n: int) -> List[List[int]]:
          matrix = [[None] * n for _ in range(n)]
          # top, right, bottom, left
          directions = ((0, 1), (1, 0), (0, -1), (-1, 0))

          r = c = d = 0
          val = 1
          for _ in range(n*n):
              matrix[r][c] = val
              val += 1

              cr, cc = r + directions[d][0], c + directions[d][1]
              if 0 <= cr < n and 0 <= cc < n and matrix[cr][cc] is None:
                  r, c = cr, cc
              else:
                  d = (d + 1) % 4
                  r, c = r + directions[d][0], c + directions[d][1]

          return matrix
        ```
     * Approach2: Layer by Layer, Time:O(mn), Space:O(1)
       * Python
        ```python
        def generateMatrix(self, n: int) -> List[List[int]]:
          def spiral_coords(r1, c1, r2, c2):
              # Top, go right, c1 to c2
              for c in range(c1, c2+1):
                  yield r1, c

              # Right, go down , r1+1 to r2
              for r in range(r1 + 1, r2 + 1):
                  yield r, c2

              # prevent duplicate for single row and single col
              if r1 < r2 and c1 < c2:
                  # Bottom, go left, c2-1 to c1
                  for c in range(c2-1, c1-1, -1):
                      yield r2, c

                  # Left, go up  r2-1 to r1-1
                  for r in range(r2-1, r1, -1):
                      yield r, c1

          # top left corner
          r1 = c1 = 0
          # bottom right corner
          r2, c2 = n-1, n-1

          val = 1
          matrix = [[None] * n for _ in range(n)]

          while r1 <= r2 and c1 <= c2:
              for r, c in spiral_coords(r1, c1, r2, c2):
                  matrix[r][c] = val
                  val += 1

              r1, c1 = r1 + 1, c1 + 1
              r2, c2 = r2 - 1, c2 - 1

          return matrix
        ```
 * **Search**:
   * 074:	Search a 2D Matrix (M)
     * Definitions:
       * Integers in each row are sorted from left to right.
       * **The first integer of each row is greater than the last integer of the previous row.**
     * Approach1: Brute Force, Time:O(mn), Space:O(1)
     * Approach2: Search Space Reduction, O(m+n), Space:O(1)
       * Starting from bottom left of the matrix (or from top right ?)
         * if the target is greater than the cur val, go right
         * if the target is smaller than the cur val, go up
       * Python
        ```python
        def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
          if not matrix:
              return False

          row, col = len(matrix), len(matrix[0])

          if not col:
              return False

          found = False
          r, c = row - 1, 0

          while r >= 0 and c < col:
              if target == matrix[r][c]:
                  found = True
                  break
              # truncate the col
              elif target > matrix[r][c]:
                  c += 1
              else:
                  r -= 1

          return found
        ```
     * Approach3: Twice Binary Search, Time:O(log(m)+log(n)) = O(log(mn)) , Space:O(1)
       * Determine the row
         * Boundaries:
           * Left Boundary
             * val before left boundary is smaller than the target
           * Right Boundary
             * val after right boundary is greater than the target
         * If not found:
           * return right is right is not -1, A[right] < target < A[left]
       * Determine the col, standard binary search
       * Python
          ```python
          def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            def bin_search(row, target):
                l, r = 0, len(row)-1
                res = False
                while l <= r:
                    mid = (l + r) // 2
                    if target == row[mid]:
                        res = True
                        break
                    elif target > row[mid]:
                        l = mid + 1
                    else:
                        r = mid - 1
                return res

            if not matrix:
                return False

            row = len(matrix)
            col = len(matrix[0])
            # special case, only 1 row
            if row == 1:
                return bin_search(matrix[0], target)

            res = False
            # step1, determine the row
            l, r = 0, row-1
            while l <= r:
                mid = (l + r) // 2
                if target == matrix[mid][0]:
                    res = True
                    break
                elif target > matrix[mid][0]:
                    l = mid + 1
                else:
                    r = mid - 1

            # r == -1 means the target is smaller than the matrix[0][0]
            if res or r == -1:
                return res

            # step2, search the row
            return bin_search(matrix[r], target)
          ```
     * Approach4: Once Binary Search: Time:O(log(mn)) = O(log(m)+log(n)), Space:O(1)
       * matrix indices from 0 to m*n-1
       * transfer index to r, c
         * r = idx // n
         * c = idx % n
       * Python
         ```python
         def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            if not matrix:
                return False

            row, col = len(matrix), len(matrix[0])
            res = False
            # step1, determine the row
            left, right = 0, row * col - 1
            while left <= right:
                mid = (left + right) // 2

                r, c = mid // col, mid % col

                if target == matrix[r][c]:
                    res = True
                    break

                elif target > matrix[r][c]:
                    left = mid + 1
                else:
                    right = mid - 1

            return res
         ```
   * 240:	Search a 2D Matrix II (M)
     * Definitions
       * Integers in each row are sorted in ascending from left to right.
       * **Integers in each column are sorted in ascending from top to bottom.**
     * Error Approach: Binary Search, O(logm + logn)
       * Search in first row to determine the column
         * Seach the column
       * Search the first col to determine the row
         * Search the row
       * **Error Solution**, consider the case
       * [ [11,13,15,17,19],
           [12,14,16,18,20]
         ], and we want to find 13
     * Approach1: Brute Force, Time:O(mn), Space:O(1)
     * Approach2: Binary Search in each row:, Time:O(mlogn), Space:O(1)
       * Python
        ```python
        def searchMatrix(self, matrix, target):
          def binary_search_row(row):

              left, right = 0, col-1
              found = False

              while left <= right:
                  mid = (left + right) // 2
                  if target == matrix[r][mid]:
                      found = True
                      break
                  elif target > matrix[r][mid]:
                      left = mid + 1
                  else:
                      right = mid - 1

              return found

          if not matrix:
              return False

          row, col = len(matrix), len(matrix[0])

          if not col:
              return False

          found = False

          for r in range(row):
              if target == matrix[r][0]:
                  found = True
                  break
              elif target > matrix[r][0]:
                  found = binary_search_row(r)
                  if found:
                      break
              else: # target < matrix[r][0]
                  break

          return found
        ```
     * Approach3: Search Space Reduction, Time:O(m+n), Space:O(1)
       * Starting from bottom left of the matrix (or from top right ?)
       * if the target is greater than the cur val, go right
       * if the target is smaller than the cur val, go up
       * Python
        ```python
        def searchMatrix(self, matrix, target):
          if not matrix:
              return False

          row, col = len(matrix), len(matrix[0])

          if not col:
              return False

          found = False

          r = row - 1
          c = 0
          while r >= 0 and c < col:
              if target == matrix[r][c]:
                  found = True
                  break
              # truncate the col
              elif target > matrix[r][c]:
                  c += 1
              # truncate the row
              else:
                  r -= 1

          return found
        ```
     * Approach4: Divide and Conquer, Time:O(nlogn)(not sure), Space:O(log(n))
       * Divide the matrix into four parts.
       * Ref:
         * https://leetcode.com/articles/search-a-2d-matrix-ii/
         * https://leetcode.com/problems/search-a-2d-matrix-ii/discuss/66154/Is-there's-a-O(log(m)%2Blog(n))-solution-I-know-O(n%2Bm)-and-O(m*log(n))
       * Time:
         * How to Evaluate the time complexity ??
       * (r1, c1): top left corner
       * (r2, c2): bottom right corner
       * For each matrix
         * Check if the target is in the boundary of the matrix
           * matrix[r1][c1] <= target <= matrix[r1][c1]
         * Divide the matrix into 4 part
           * mid rol = (c1 + c2) // 2
             * Why use (c1 + c2) // 2 as middle row ?
               * This is a proper position to skip half of matrix (left up + bottom right)
           * mid col = the first val in the mid rol which is greater than the target
           * Top-left and bottom-right submatries can be skipped
             * val in top-left submatrix < target
             * val in bottom-right submatrix > target
           * Iterative top-right and bottom-left submatrices
       * Python
         ```python
         def searchMatrix(self, matrix, target):
            def bin_search_row(col_num, left, right):
                found = False
                while left <= right:
                    mid = (left + right) // 2

                    if target == matrix[mid][col_num]:
                        found = True
                        break
                    elif target > matrix[mid][col_num]:
                        left = mid + 1
                    else:
                        right = mid - 1

                return found, left

          if not matrix:
              return False

          row, col = len(matrix), len(matrix[0])

          if not col:
              return False

          found = False

          # top-left corner
          r1 = c1 = 0
          # bottom-right corner
          r2, c2 = row-1, col-1

          stack = [(r1, c1, r2, c2)]

          found = False

          while stack:
              r1, c1, r2, c2 = stack.pop()

              # out of boundary
              if r1 > r2 or c1 > c2:
                  continue

              # check if the target is in the boundary of the matrix
              if target < matrix[r1][c1] or taret > matrix[r2][c2]:
                  continue

              mid_c = (c1 + c2) // 2
              found, mid_r = bin_search_row(mid_c, r1, r2)

              if found:
                  break

              # bottom left
              stack.append((mid_r, c1, r2, mid_c-1))

              # top right
              stack.append((r1, mid_c+1, mid_r-1, c2))

          return found
         ```
 * 378:	**Kth Smallest Element** in a Sorted Matrix (M)
   * Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.
   * Approach1: Brute Force, Time:O(nm), Space:O(k)
     * Python
      ```python
      class Element(object):
        def __init__(self, val):
            self.val = val

        def __lt__(self, other):
            return self.val > other.val

      def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
          row, col = len(matrix), len(matrix[0])
          heap = []

          for r in range(row):
              for c in range(col):
                  if len(heap) < k:
                      heapq.heappush(heap, Element(matrix[r][c]))
                  elif matrix[r][c] < heap[0].val:
                      heapq.heapreplace(heap, Element(matrix[r][c]))
                  else:
                      break

          return heap[0].val
      ```
   * Approach2: Simulation of merge k sorted list, Time:O(klog((min(k, n))), Space:O(min(k, n))
     * Assume that each row is a sorted linked list, val in in first col is the head
     * Python
      ```python
      class Element(object):
        def __init__(self, r, c, val):
            self.r = r
            self.c = c
            self.val = val

        def __lt__(self, other):
            return self.val < other.val

        def __repr__(self):
            return str(self.val)

      def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
          row, col = len(matrix), len(matrix[0])
          heap = []

          # reduce complexity of heapify from mlogm to m
          # we don't need to push more than 'k' elements in the heap
          # since the val in those row is greater than matrix[k][0]
          for r in range(min(k, row)):
            heap.append(Element(r, 0, matrix[r][0]))
          heapq.heapify(heap)

          cnt = 0
          kth_min = None
          while heap and cnt < k:
              r, c, kth_min = heap[0].r, heap[0].c, heap[0].val
              cnt += 1

              if c + 1 < col:
                  heapq.heapreplace(heap, Element(r, c+1, matrix[r][c+1]))
              else:
                  heapq.heappop(heap)

          return kth_min
      ```
   * Approach3: Binary Search, Time:O(log(mn)*(m+n)), Space:O(1)
     * Ref:
       * https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/301357/Simple-to-understand-solutions-using-Heap-and-Binary-Search-JavaPython
     * Algo
       * pick a middle
       * for each row, cnt the value which is smaller than the middle
         * inner loop should change to binary search as well
       * if cnt != k, update the middle and try again
     * Time:
       * Total: O(mn) iteration, each iteration takes O(nlogn)
     * Python
      ```python
      def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        def count_smaller_euqal_val(target):
            max_smaller = matrix[0][0]
            min_larger = matrix[row-1][col-1]
            cnt = 0
            r, c = row-1, 0
            # Search Space Reduction
            while r >= 0 and c < col:
                if matrix[r][c] <= target:
                    max_smaller = max(max_smaller, matrix[r][c])
                    cnt += (r + 1)
                    c += 1
                else:
                    min_larger = min(min_larger, matrix[r][c])
                    r -= 1

            return cnt, max_smaller, min_larger

        row, col = len(matrix), len(matrix[0])
        lo = matrix[0][0]
        hi = matrix[row-1][col-1]
        while lo <= hi:

            if lo == hi: # <--- since hi may be the middle, we need to break here
              break

            mid = (lo + hi) // 2
            cnt, max_smaller, min_larger = count_smaller_euqal_val(mid)
            if cnt < k:
                # min_larger > mid
                lo = min_larger
            else:  # cnt >= k
                # max_smaller <= mid -> that is, hi may be the mid
                hi = max_smaller

        k_smallest = lo
        return k_smallest
      ```
   * Approach4: Time:O(m)
     * m is the row
     * https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85170/O(n)-from-paper.-Yes-O(rows).
 * 073:	Set Matrix Zeroes (M)
   * Approach1: Addititionl Memory, Time:O(mn), Space:O(m+n)
     * Python
      ```python
      def setZeroes(self, matrix: List[List[int]]) -> None:
        def set_row_zero(r):
            for c in range(col):
                matrix[r][c] = 0

        def set_col_zero(c):
            for r in range(row):
                matrix[r][c] = 0

        if not matrix:
            return

        row, col = len(matrix), len(matrix[0])

        if not col:
            return

        zero_r = set()
        zero_c = set()

        for r in range(row):
            for c in range(col):
                if matrix[r][c] == 0:
                    zero_r.add(r)
                    zero_c.add(c)

        for r in zero_r:
            set_row_zero(r)

        for c in zero_c:
            set_col_zero(c)
      ```
   * Approach2: In place, Time:O(mn), Space:O(1)
     * Python
        ```python
        def setZeroes(self, matrix: List[List[int]]) -> None:
          def set_row_zero(r):
              for c in range(col):
                  matrix[r][c] = 0

          def set_col_zero(c):
              for r in range(row):
                  matrix[r][c] = 0

          if not matrix:
              return

          row, col = len(matrix), len(matrix[0])

          if not col:
              return

          zero_first_row = zero_first_col = False

          # check first row
          for c in range(col):
              if matrix[0][c] == 0:
                  zero_first_row = True
                  break

          # check first col
          for r in range(row):
              if matrix[r][0] == 0:
                  zero_first_col = True
                  break

          # use first row and col as memo
          for r in range(1, row):
              for c in range(1, col):
                  if matrix[r][c] == 0:
                      matrix[0][c] = 0
                      matrix[r][0] = 0

          # check first col to zero row
          for r in range(1, row):
              if matrix[r][0] == 0:
                  set_row_zero(r)

          # check first row to zero col
          for c in range(1, col):
              if matrix[0][c] == 0:
                  set_col_zero(c)

          if zero_first_col:
              set_col_zero(0)

          if zero_first_row:
              set_row_zero(0)
        ```
 * 311:	**Sparse** Matrix Multiplication (M)
   * Approach1: Brute Force, Time:(mnk), Space:O(1)
     * Python
      ```python
      def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:

        def product(row, col, cnt):
            res = 0
            for i in range(cnt):
                res += A[row][i] * B[i][col]
            return res

        r1, c1 = len(A), len(A[0])
        r2, c2 = len(B), len(B[0])
        r3, c3 = r1, c2

        res = [[0 for _ in range(c3)] for _ in range(r3)]

        for r in range(r3):
            for c in range(c3):
                res[r][c] = product(r, c, c1)
        return res
      ```
   * Approach2: Keep info of zero row and zero col Time:(mnk), Space:O(m + n)
     * Python
      ```python
      def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        def product(row, col, cnt):
            if zero_row_a[row] and zero_col_b[col]:
                return 0

            res = 0
            for i in range(cnt):
                res += A[row][i] * B[i][col]
            return res

        r1, c1 = len(A), len(A[0])
        r2, c2 = len(B), len(B[0])
        r3, c3 = r1, c2

        res = [[0 for _ in range(c3)] for _ in range(r3)]

        zero_row_a = [None] * r1
        zero_col_b = [None] * c2

        for r in range(r1):
            is_zero_row = True
            for c in range(c1):
                if A[r][c] != 0:
                    is_zero_row = False
                    break
            zero_row_a[r] = is_zero_row

        for c in range(c2):
            is_zero_col = True
            for r in range(r2):
                if B[r][c] != 0:
                    is_zero_col = False
                    break
            zero_col_b[c] = is_zero_col

        for r in range(r3):
            for c in range(c3):
                res[r][c] = product(r, c, c1)

        return res
      ```
   * Approach3: Sum and accumulate, Time:(mnk), Space:O(1)
     * Ref:
       * https://leetcode.com/problems/sparse-matrix-multiplication/discuss/76151/54ms-Detailed-Summary-of-Easiest-JAVA-solutions-Beating-99.9
     * **The key part of smart solution is that: it does not calculate the final result at once, and it takes each value from A, and calculate and partial sum and accumulate it into the final spot**.
     * For example, for each value A[i][k], if it is not zero, it will be used at most nB times ( n is B[0].length ), which can be illustrated as follow:
     * Generally for the following equation:
        ```txt
        C[i][0] = A[i][0]*B[0][0]  + A[i][1]*B[1][0] + A[i][2]*B[2][0] + ... A[i][k]B[k][0] .... A[i][K]*B[K][0]
        C[i][1] = A[i][0]*B[0][1]  + A[i][1]*B[1][1] + A[i][2]*B[2][1] + ... A[i][k]B[k][0] .... A[i][K]*B[K][1]
        ...
        C[i][nB]= A[i][0]*B[0][nB] + A[i][1]*B[1][nB] + A[i][2]*B[2][nB] + ... A[i][k]B[k][nB] .... A[i][K]*B[K][nB]
        ```
      * Python
        ```python
        def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
          r1, c1 = len(A), len(A[0])
          r2, c2 = len(B), len(B[0])
          r3, c3 = r1, c2

          res = [[0 for _ in range(c3)] for _ in range(r3)]

          for i in range(r1):
              for k in range(c1):

                  if A[i][k] == 0:
                      continue

                  for j in range(c2):
                      if B[k][j] != 0:
                          res[i][j] += A[i][k] * B[k][j]

          return res
        ```
 * 370:	Range Addition (M)
 * 296:	Best Meeting Point (H)
 * 361: Bomb Enemy (M)
 * 317: Shortest Distance from All Buildings (H)
 * 302: Smallest Rectangle Enclosing Black Pixels (H)
 * 036: Valid Sudoku (M)
 * 037: Sudoku Solver (H)
### Binary Search
  * Tips:
    * Identify the definition of the left boundary and right boundary.
    * Identify the result when the target is not found within the boundaries.
    * Consider the cases that have one element and two elements.
  * 374: Guess Number Higher or Lower (E)
    * Approach1: Linear Search, Time:O(n), Space:O(1)
    * Approach2: Binary Search, Time:O(logn), Space:O(1)
      * Standard Binary Search
      * boundaries
        * left boundary: The values before left boundary are less than the target.
        * right boundary: the values after right boundary are greater than the target.
      * If the target not found:
        * return not_found (-1)
      * Python
        ```python
        def guessNumber(self, n):
          res = not_found = -1
          left, right = 1, n

          while left <= right:

              mid = (left + right) // 2

              guess_res = guess(mid)
              if guess_res == 0:    # got it
                  res = mid
                  break
              elif guess_res == 1:  # guess number is higher, seach right
                  left = mid + 1
              else :                # guess number is lower , seach left
                  right = mid - 1

          return res
          ```
  * 035: Search Insert Position (E)
    * Find the first value >= target
    * Approach1: Linear, Time:O(n), Space:O(1)
    * Approach2: Binary Search, Time:O(logn), Space:O(1)
      * Boundaries:
        * left boundary:
          * Value before the left boundary are smaller than the target
        * right boundary:
          * Value after the right boundary are bigger than the target
      * If the target not found:
        * return left, which is the position of the first number > target
      * Python
        ```python
        def searchInsert(self, nums: List[int], target: int) -> int:
          if not nums:
              return 0

          # find the first value >= target
          left, right = 0, len(nums)-1
          res = not_found = -1

          while left <= right:

              mid = (left + right) // 2

              if target == nums[mid]:
                  res = mid
                  break
              elif target < nums[mid]:
                  right = mid - 1
              else:
                  left = mid + 1

          if res is not_found:
              res = left

          return res
        ```
  * 278: First Bad Version (E)
    * Definitions
      * Assume there exists a bad version
    * Approach1: linear search, Time:O(n), Space:O(1)
    * Approach2: binary search, Time:O(logn), Space:O(1)
      * boundaries:
        * left boundary:
          * Versions before left are good
        * right boundary:
          * Versions after right are bad
      * If the target is not found:
        * It's impossible according to the definiton
        * Python
          ```python
          def firstBadVersion(self, n):
            # assume there exists a bad version
            if n == 1:
                return 1

            # if the first version is bad, just return
            if isBadVersion(1):
                return 1

            left = 2   # versions before left are good
            right = n  # versions after right are bad
            while left <= right:
                mid = (left + right) // 2
                if isBadVersion(mid):
                    right = mid - 1
                else:
                    left = mid + 1
            return left
          ```
    * 162: Find **Peak Element** (M)
      * 3 cases:
        * case1: Peak is the first element
          * All the number appear in a descending order.
        * case2: Peak is the last element
          * All the number appear in a ascending order.
        * case3: Peak is in the middle of the nums
      * Approach1: Linear, Time:O(n), Space:O(1)
        * Python
          ```python
          def findPeakElement(self, nums):
            n = len(nums)

            if n == 1:
                return 0

            peak = n - 1
            # from 0 to n-2
            for i in range(0, n-1):
                if nums[i] > nums[i+1]:
                    peak = i
                    break

            return peak
          ```
      * Approach2: Binary Search, Time:O(logn), Space:O(1)
        * Peak is not unique, we can return any peak in the nums
        * Python
          ```python
          def findPeakElement(self, nums):
            n = len(nums)

            if n == 1:
                return 0

            left = 0
            right = n-1
            peak = None
            while left <= right:

                mid = (left + right) // 2

                if mid == n-1:
                    peak = n-1
                    break
                # descending order, find left part
                if nums[mid] > nums[mid+1]:
                    right = mid - 1
                # ascending order, find the right part
                else:
                    left = mid + 1

            if peak is None:
                peak = left

            return peak
          ```
  * 034: Find **First and Last Position of Element** in Sorted Array (M)
      * Approach1: Linear Search, Time:O(n), Space:O(1)
        * Python
          ```python
          def searchRange(self, nums: List[int], target: int) -> List[int]:
            not_found = -1
            start = end = not_found
            for idx, val in enumerate(nums):
                if val == target:
                    if start == not_found:
                        start = idx
                    end = idx

            return [start, end]
          ```
      * Approach2: Binary Search, Time:O(n), Space:O(1)
        * Boundaries:
          * to find start
            * left: val before left boundary is less than the target
            * right: val after right boundary is **grater or equal** to the target
          * to find end
            * left: val before left boundary is **less or equal** to the target
            * right: val after right boundary is grater than the target
        * Python
          ```python
          def searchRange(self, nums: List[int], target: int) -> List[int]:
            def bin_search_left(left, right):
                start = not_found
                l, r = left, right

                while l <= r:
                    mid = (l + r) // 2
                    # val after right boundary is **grater or equal** to the target
                    if target <= nums[mid]:
                        r = mid - 1
                    # target > nums[mid] -> nums[mid] < target
                    else:
                        l = mid + 1

                if left <= l <= right and nums[l] == target:
                    start = l

                return start

            def bin_search_right(left, right):
                end = not_found
                l, r = left, right

                while l <= r:
                    mid = (l + r) // 2
                    # val before left boundary is **less or equal** to the target
                    if target >= nums[mid]:
                        l = mid + 1
                    else: # target < nums[mid] -> mid > target
                        r = mid - 1

                if left <= r <= right and nums[r] == target:
                    end = r

                return end

            not_found = -1
            # find start
            left = 0
            right = len(nums) - 1

            s = bin_search_left(left, right)
            if s == not_found:
                e = not_found
            else:
                e = bin_search_right(s, right)

            return [s, e]
          ```
  * 1150: Check If a Number Is Majority Element in a Sorted Array (E)
  * 275: H-Index II (M)
    * Approach1: Linear search: O(n)
      * Concept
        * Sort the citations array in **ascending order** (draw it).
        * c = citations[i]. We would know that the number of articles whose citation number is higher than c would be n - i - 1.
        * And together with the current article, **there are n - i articles that are cited at least c times**.
      * Python
        ```python
        def hIndex(self, citations: List[int]) -> int:
          res = 0
          max_hidx = len(citations)

          for idx, cita in enumerate(citations):
              h_idx = max_hidx - idx
              if cita >= h_idx:
                  res = h_idx
                  break

          return res
        ```
    * Approach2: **Binary Search**: O(log(n))
      * Search algo like 35
      * Ref:
        * https://leetcode.com/problems/h-index-ii/discuss/71063/Standard-binary-search
        * https://leetcode.com/problems/h-index-ii/solution/
      * Boundaries
        * Indices after right boundary are accetable.
        * Indices before left boundary are unaccetable.
        * example:
          ```txt
          val        3  4  5  6  7
          idx        0  1  2  3  4
          max-idx  6 5  4  3  2  1  0
                     l           r
          ```
      * When the target is not found in the range
        * return (max_cita - left)
        * There are 2 ending cases
          * case1 (left, **right, new_left**)
            * Left jumps right, and new_left is greater than the right boundary.
            * According to the definition, the result should be **new_left**
          * case2 (**new_right, left**, right)
            * Right jumps left, and new_right is smaller than the left boundary.
            * According to the definition, the result should be **left**
      * Python
        ```python
        def hIndex(self, citations: List[int]) -> int:
          max_hidx = len(citations)
          left, right = 0, len(citations)-1

          idx = None
          while left <= right:
              mid = (left + right) // 2
              h_idx = max_hidx - mid

              if citations[mid] == h_idx:
                  idx = mid
                  break

              elif citations[mid] < h_idx:
                  left = mid + 1

              else:
                  right = mid - 1

          if idx is None:
              idx = left

          return max_hidx - idx
        ```
  * 033 & 081: Search in **Rotated Sorted Array** (M)
    * 033: unique
    * 081: allow duplicates
    * For duplicates, the key is how to handle the case like:
      ```txt
         nums  [4, 4, 1, 2, 3, 4, 4 ]
         idx    0  1  2  3  4  5  6
                l  p           p* r
         In such case, if we choose a pivot with value 4
         we can not know the pivot is in the left part (p) or right part (p*)
         if in the left part, nums[l:p] is ascending order
         if in the right part, nums[p:r] is ascending order
         one solution is to move l to right until nums[left] != nums[right]
      ```
    * Approach1: Linear Search, Time:O(n), Space:O(1)
    * Approach2: Get Rotate Index (unique value), Time:O(logn), Space:O(1)
      * Algo:
        * step1: Find the rotate index, this index split the array to 2 parts.
          * **the (rotate index - 1) is the last element in the left part.**
          * There are 2 cases
            * case1: [0, 1, 2, 3, 4, 5] -> rotate idx should be 0
            * case2: [3, 4, 5, 0, 1, 2] -> rotate idx should be 3
        * step2: Identify which part has to look for target
        * step3: Binary seach in the part
      * When the target is not found:
        * The result should be the first index, since the array is non-rotated.
      * Python
        ```python
        def search(self, nums: List[int], target: int) -> int:
          def find_rotate_idx(left, right):
              """
                * There are 2 cases
                * case1: [0, 1, 2, 3, 4, 5] -> rotate idx should be 0
                * case2: [3, 4, 5, 0, 1, 2] -> rotate idx should be 3
              """
              right_boundary = right
              rotate_idx = 0

              while left <= right:

                  mid = (left + right) // 2
                  # case1
                  if mid == right_boundary:
                      break

                  # case2, no rotate idx found
                  if nums[mid] > nums[mid+1]:
                      rotate_idx = mid + 1
                      break

                  else:
                      # mid is in the left part, find the last element of the left part
                      if nums[mid] >= nums[left]:
                          left = mid + 1
                      # mid is in the right part, try to go to left part
                      else:
                          right = mid - 1

              return rotate_idx

          def binary_search(left, right):
              res = not_found

              while left <= right:
                  mid = (left + right) // 2

                  if target == nums[mid]:
                      res = mid
                      break

                  elif target < nums[mid]:
                      right = mid - 1
                  else:
                      left = mid + 1

              return res

          not_found = -1

          n = len(nums)

          if n < 1:
              return not_found

          if n == 1:
              return 0 if nums[0] == target else not_found

          rotate_idx = find_rotate_idx(0, n-1)

          res = not_found
          if rotate_idx == 0:
              res = binary_search(0, n-1)
          else:
              # find left part
              if target >= nums[0]:
                  res = binary_search(0, rotate_idx-1)
              # find right part
              else:
                  res = binary_search(rotate_idx, n-1)

          return res
        ```
    * Approach3: Get Rotate Index (duplicated value), Time:O(logn)~O(n), Space:O(1)
      * Fix the case [2,2,2,0,1,2,2,2], can not find the correct rotate idx
      * Python
        ```python
        def search(self, nums: List[int], target: int) -> bool:
          def find_rotate_idx(left, right):
              """
              * There are 2 cases
              * case1: [0, 1, 2, 3, 4, 5] -> rotate idx should be 0
              * case2: [3, 4, 5, 0, 1, 2] -> rotate idx should be 3
              """
              # Previous algorithm can deal with duplicated items
              # if the head of array is not a duplicated item
              while left != right and nums[left] == nums[right]:
                  left += 1

              rotate_idx = left
              right_boundary = right

              while left <= right:
                  mid = (left + right)//2

                  # case1:
                  if mid == right_boundary:
                      break

                  # case2:
                  if nums[mid] > nums[mid+1]:
                      rotate_idx = mid + 1
                      break
                  else:
                      # in the left part
                      if nums[mid] >= nums[left]:
                          left = mid + 1
                      else:
                          right = mid - 1

              return rotate_idx

          def binary_search(left, right):
              is_found = False

              while left <= right:
                  mid = (left + right) // 2

                  if target == nums[mid]:
                      is_found = True
                      break

                  elif target <= nums[mid]:
                      right = mid - 1

                  else:
                      left = mid + 1

              return is_found

          n = len(nums)

          if n < 1:
              return False

          if n == 1:
              return True if nums[0] == target else False

          is_found = False

          rotate_idx = find_rotate_idx(0, n-1)

          if rotate_idx == 0:
              is_found= binary_search(0, n-1)
          else:
              if target >= nums[0]:
                  is_found = binary_search(0, rotate_idx-1)
              else:
                  is_found = binary_search(rotate_idx, n-1)

          return is_found
        ```
    * Approach4: One Pass binary Search (unique value), Time:O(logn), Space:O(1)
      * The key idea is to find the non-roated array
        * example:
          * case1: left  pivot rotate_idx right
            * left to pivot is un-rotated part
          * case2: left rotate_idx  pivot  right
            * pivot to right is un-rotated part
      * Python
        ```python
        def search(self, nums: List[int], target: int) -> int:
          not_found = -1
          n = len(nums)

          if n < 1:
              return not_found

          if n == 1:
              return 0 if nums[0] == target else not_found

          res = not_found
          left = 0
          right = n-1
          while left <= right:
              mid = (left + right) // 2
              if nums[mid] == target:
                  res = mid
                  break
              else:
                  # don't forget to add eeuql(=)
                  if nums[mid] >= nums[left]:
                      # case1: left to mid is non-roated array
                      if nums[left] <= target < nums[mid]:
                          right = mid - 1
                      else:
                          left = mid + 1
                  else:
                      # case2: mid to right is non-rotated array
                      if nums[mid] < target <= nums[right]:
                          left = mid + 1
                      else:
                          right = mid - 1
          return res
        ```
    * Approach5: One Pass binary Search (duplicated value), Time:O(logn)~O(n), Space:O(1)
      * Python
        ```python
        def search(self, nums: List[int], target: int) -> bool:
          n = len(nums)

          if n < 1:
              return False

          if n == 1:
              return True if nums[0] == target else False

          is_found = False
          left = 0
          right = n-1

          # Previous algorithm can deal with duplicated items
          # if the head of array is not a duplicated item
          # special case: [4, 4, 1, 2, 3, 4, 4]
          while left != right and nums[left] == nums[right]:
              left += 1

            while left <= right:

                mid = (left + right) // 2

                if nums[mid] == target:
                    is_found = True
                    break

                else:
                    if nums[mid] >= nums[left]:
                        # left to mid is an un-rotated array
                        if nums[left] <= target < nums[mid]:
                            right = mid - 1
                        else:
                            left = mid + 1
                    else:
                        #  mid to right is an un-rotated array
                        if nums[mid] < target <= nums[right]:
                            left = mid + 1
                        else:
                            right = mid - 1

            return is_found
        ```
  * 153 & 154: Find Minimum in **Rotated Sorted Array** (M)
    * The same concept like 33 and 81.
    * 153: unique
    * 154: allow duplicates
    * Approach1: Linear Search, Time:O(n), Space:O(1)
    * Approach2: Find Rotate index (unique value), Time:O(logn), Space:O(1)
      * Python
        ```python
        def findMin(self, nums: List[int]) -> int:
          def find_rotate_idx(left, right):
              rotate_idx = 0
              right_boundary = right

              while left <= right:
                  mid = (left + right) // 2

                  if mid == right_boundary:
                      break

                  if nums[mid] > nums[mid+1]:
                      rotate_idx = mid + 1
                      break

                  else:
                      if nums[mid] >= nums[left]:
                          left = mid + 1
                      else:
                          right = mid - 1

              return rotate_idx

            n = len(nums)

            if n < 1:
                return None

            if n == 1:
                return nums[0]

            rotate_idx = find_rotate_idx(0, n-1)

            return nums[rotate_idx]
        ```
    * Approach3: Find Rotate index (duplicated value), Time:O(logn~n), Space:O(1)
      * Python
        ```python
        def findMin(self, nums: List[int]) -> int:
          def find_rotate_idx(left, right):
              # Previous algorithm can deal with duplicated items
              # if the head of array is not a duplicated item
              while left != right and nums[left] == nums[right]:
                  left += 1

              rotate_idx = left
              right_boundary = right

              while left <= right:
                  mid = (left + right) // 2

                  if mid == right_boundary:
                      break

                  if nums[mid] > nums[mid+1]:
                      rotate_idx = mid + 1
                      break

                  else:
                      if nums[mid] >= nums[left]:
                          left = mid + 1
                      else:
                          right = mid - 1

              return rotate_idx

          n = len(nums)

          if n < 1:
              return None

          if n == 1:
              return nums[0]

          rotate_idx = find_rotate_idx(0, n-1)

          return nums[rotate_idx]
        ```
  * 222: Count **Complete** Tree Nodes (M)
    * Approach1: Preorder, Time:O(n), Space:O(n)
      * Python
        ```python
        def countNodes(self, root: TreeNode) -> int:
          if not root:
              return 0

          node_cnt = 0
          stack = [root]

          while stack:
              node = stack.pop()
              node_cnt += 1

              if node.left:
                  stack.append(node.left)

              if node.right:
                  stack.append(node.right)

          return node_cnt
        ```
    * Approach2: Binary Search, Time:O(d^2), Space:O(1)
      * Ref:
        * https://leetcode.com/articles/count-complete-tree-nodes/
      * Time:
        * Find depth: O(d)
        * Check node cnt in the last level: O(d^2)
        * Total cost: O(d^2) = O(log(n)^2) (complete binary tree)
      * Definitions:
        * Assume depth starts from 0
        * Total nodes is 2^d -1 + last_level_node
        * The maximum node cnt in the last level is 2^d
      * Boundaries:
        * left boundary:
          * The nodes before the left boundary exist.
        * right boundary:
          * The are no nodes after the right boundary.
      * Python
        ```python
        def countNodes(self, root: TreeNode) -> int:

          def get_depth(cur):
              depth = 0

              while cur.left:
                  depth += 1
                  cur = cur.left

              return depth

          def check_node_exists(node_idx, d, cur):
              target = cur
              # go at most depth level
              # Last level nodes are enumerated from 0 to 2**d -1
              left = 0
              right = 2**d - 1
              for _ in range(d):
                  mid = (left + right) // 2
                  # mid is inclued in the left part
                  if node_idx <= mid:
                      target = target.left
                      right = mid - 1
                  else:
                      target = target.right
                      left = mid + 1

              return target is not None

          if not root:
              return 0

          d = get_depth(root)

          if d == 0:
              return 1

          # Last level nodes are enumerated from 0 to 2**d -1
          # we starts from 2nd node (index 1), since 1st node always exists.
          left, right = 1, 2**d-1
          while left <= right:
              mid = (left + right) // 2

              if check_node_exists(mid, d, root):
                  # every nodes before mid exist
                  left = mid + 1
              else:
                  # there are no nodes after mid
                  right = mid - 1

          # The tree contains 2**d - 1 nodes on the first (d - 1) levels
          # and left nodes on the last level.
          return (2**d - 1) + left
        ```
  * 004: **Median** of **Two Sorted Arrays** (H)
    * Approach1: Merge and Find, Time:O(m+n), Space:O(m+n)
      * Time: O(m+n)
        * Merge takes O(m+n)
      * Space: O(m+n)
        * Extra space to keep merged array
    * Approach2: Binary Search, Time:O(log(min(m,n))), Space:O(1)
      * Ref:
        * [Concept](https://www.youtube.com/watch?v=LPFhl65R7ww)
        * [Implementation](https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2481/Share-my-O(log(min(mn)))-solution-with-explanation)
      * ![median_of_2](./image/algo/median_of_two.png)
      * Python Solution:
        ```python
        MAX_VAL = float('inf')
        MIN_VAL = -float('inf')

        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            n1 = len(nums1)
            n2 = len(nums2)

            if n1 > n2:
                # keep n1 shorter
                nums1, nums2 = nums2, nums1
                n1, n2 = n2, n1

            """
            This number means how many number is smaller than the median:
            0:  0 elements are smaller than the median
            n1: all elements are smaller than the median
            ( total n + 1 cases starting from 0 to n )
            """
            left_b = 0    # left boundary
            right_b = n1  # right boundary
            """
            The half len of merged array (merge nums1 and nums2)
            +1 to let left side of the merged array has more than one element when the length is odd.
            p_n1 + p_n2 = (n1 + n2 + 1) // 2
            """
            half_len = (n1 + n2 + 1) // 2

            while left_b <= right_b:
                p_n1 = (left_b + right_b)//2
                # p_n1 + p_n2 = (n1 + n2 + 1) // 2
                p_n2 = half_len - p_n1
                """
                p_n1 == 0 means nothing is there on left side. Use -INF as sentinel
                p_n1 == n1 means there is nothing on right side. Use +INF as sentinel
                for example, if nums1 = [1, 2, 3]
                case0: [-INF], [1, 2, 3]  # partition_n1 == 0
                case1: [1], [2,3]
                case2: [1, 2], [3]
                case3: [1, 2, 3], [+INF]  # partition_n1 == n1
                """
                max_left_n1 = MIN_VAL if p_n1 == 0 else nums1[p_n1-1]  # case0
                min_right_n1 = MAX_VAL if p_n1 == n1 else nums1[p_n1]  # case3

                max_left_n2 = MIN_VAL if p_n2 == 0 else nums2[p_n2-1]
                min_right_n2 = MAX_VAL if p_n2 == n2 else nums2[p_n2]

                if (max_left_n1 <= min_right_n2 and max_left_n2 <= min_right_n1):
                    if (n1 + n2) % 2 == 0:  # even number of elements
                        return (max(max_left_n1, max_left_n2) + min(min_right_n1, min_right_n2))/2
                    else:
                        return float(max(max_left_n1, max_left_n2))

                elif max_left_n1 > min_right_n2:
                    # shrink n1 left part
                    right_b = p_n1 - 1

                else: # max_left_n2 > min_right_n1
                    # shrink n2 left part, that is, increase n1 left part
                    left_b = p_n1 + 1
            ```

      * memo[i] is the longest increasing subsequece of nums[:i+1]
        * possible prev indices are from 0 to i
      * Python
        ```python
        def lengthOfLIS(self, nums: List[int]) -> int:
          if not nums:
              return 0

          n = len(nums)
          memo = [0 for _ in range(n)]
          max_lis = memo[0] = 1

          for cur in range(1, n):
              cur_lis = 1
              for prev in range(0, cur):
                  if nums[cur] > nums[prev]:
                      cur_lis = max(cur_lis, 1+memo[prev])
              memo[cur] = cur_lis

              max_lis = max(max_lis, memo[cur])

          return max_lis
        ```
  * 315: Count of Smaller Numbers After Self (H)
  * 354: Russian Doll Envelopes (H)
### Linked List
* **Techiniques**:
  * The "**Runner**"
    * The runner techinique means that you **iterate** through the linked list **with two pointers simultaneously**, with one head of the other.
  * The "dummy node"
    * **dummy.next** alwasy point to the head node, it very useful if the head node in the list will be changed.
  * Use **reverse** instead of **stack** for space complexity reduction.
    * However, reverse will change the data of the input, use it carefully.
* Design:
  * 707: Design Linked List (E)
* **Runner** and **Detect Circle**
  * 876: Middle of the Linked List
    * Approach1: Runner, Time:O(n), Space:O(1)
      * Python
        ```python
        def middleNode(self, head: ListNode) -> ListNode:
          if not head:
              return head

          slow = fast = head
          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next

          return slow
        ```
  * 141: Linked List **Cycle** (E)
    * Similar problems:
      * 202: Happy Number (E)
    * Approach1: The runner: Time:O(n), Space:O(1)
      * Python
          ```python
          def hasCycle(self, head: ListNode) -> bool:
            has_cycle = False
            fast = slow = head

            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next

                if fast is slow:
                    has_cycle = True
                    break

            return has_cycle
          ```
  * 142: Linked List Cycle II (M) *
    * Similar problems:
      * 287: Find the Duplicate Number (M)
    * Given a linked list, return the node **where the cycle begins**. If there is no cycle, return null.
    * The length before the cycle beginning: **d**
    * The length of the cycle: **r**
    * Approach1: Hash Table, Time:O(n), Space:O(1)
      * Python
        ```python
        def detectCycle(self, head: ListNode) -> ListNode:
          d = dict()

          runner = head
          target = None
          while runner:
              if runner in d:
                  target = runner
                  break

              d[runner] = True
              runner = runner.next

          return target
        ```
    * Approach2: Floyd's Tortoise and Hare (Cycle Detection), Time:O(n), Space:O(1)
      * Algo:
        *  Need 3 runners. Fast, slow, target
        * **The intersection of fast and slow** would be **dth node** in the cycle.
          * THe distance betwwen this intersection and the cycle beginning is **r-d**
        * **The intersection of slow and target** would be the cycle beginning.
        * r-d ≡ d (mod r)
      * Python
        ```python
        def detectCycle(self, head):
          if not head:
              return None

          fast = slow = head
          has_cycle = False

          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next
              # The intersection is dth node in the cycle
              # The distance between the cycle beginning should be r-d
              if fast is slow:
                  has_cycle = True
                  break

          if not has_cycle:
            return None

          cycle_begin = head
          while true:
              # need to check first for the case d == 0
              if cycle_begin is slow:
                  break

              cycle_begin = target.next
              slow = slow.next

          return cycle_begin
        ```
* **Remove Node**
   * 237: Delete Node in a Linked List (E)
     * waste my time....
     * Python
      ```python
        def deleteNode(self, node):
          node.val = node.next.val
          node.next = node.next.next
      ```
   * 203: Remove Linked List Elements (E)
     * Remove all elements from a linked list of integers that have value val.
     * Approach1: dummy node, Time:O(n), Space:O(1)
       * Python
          ```python
          def removeElements(self, head: ListNode, val: int) -> ListNode:
            if not head:
                return head

            cur = dummy = ListNode(0)
            dummy.next = head

            while cur.next:
                if cur.next.val == val:
                    cur.next = cur.next.next
                else:
                    cur = cur.next

            return dummy.next
          ```
   * 019: Remove Nth Node From End of List (M)
     * Given a linked list, remove the n-th node **from the end** of list and return its head.
     * Approach1: runner + dummy node, Time:O(n), Space:O(1)
       * Python
          ```python
          def removeNthFromEnd(self, head, n):
            slow = dummy = ListNode(0)
            fast = dummy.next = head

            for _ in range(n):
                fast = fast.next

            while fast:
                slow, fast = slow.next, fast.next

            # delete the target
            slow.next = slow.next.next

            return dummy.next
          ```
   * 083: Remove Duplicates from **Sorted** List (E)
     * Example:
       * Input: 1->1->2->2->3->3
       * Output: 1->2->3
     * Approach1: Time:O(n), Space:O(1)
       * Python
        ```python
        def deleteDuplicates(self, head: ListNode) -> ListNode:
          if not head:
              return None

          cur = head
          while cur.next:
              if cur.val == cur.next.val:
                  cur.next = cur.next.next
              else:
                  cur = cur.next

          return head
        ```
   * 082: Remove Duplicates from **Sorted** List II (M)
     * Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
     * Example:
       * Input: 1->2->3->3->4->4->5
       * Output: 1->2->5
     * Approach1: Dummy node and a flag, Time:O(n), Space:O(1)
       * note:
         * don't forget the last delete_from_prev
       * Python
        ```python
          def deleteDuplicates(self, head: ListNode) -> ListNode:
            if not head:
                return head

            prev = dummy = ListNode(0)
            cur = dummy.next = head
            delete_from_prev = False

            while cur and cur.next:
                if cur.val == cur.next.val:
                    cur.next = cur.next.next
                    delete_from_prev = True
                else:
                    if delete_from_prev:
                        prev.next = cur = cur.next
                        delete_from_prev = False
                    else:
                        prev, cur = cur, cur.next

            if delete_from_prev:
                prev.next = cur.next

            return dummy.next
        ```
* **Reorder**
  * 206: **Reverse** Linked List (E)
    * Approach1: Time:O(n), Space:O(1)
      * Python
       ```python
       def reverseList(self, head: ListNode) -> ListNode:\
          prev = None
          cur = head

          while cur:
              nxt = cur.next
              cur.next = prev
              prev, cur = cur, nxt

          return prev
       ```
  * 092: **Reverse** Linked List II (M)
    * From **position m to n**. Do it in **one-pass**.
    * Example:
      * Input: 1->2->3->4->5->NULL, m = 2, n = 4
      * Output: 1->4->3->2->5->NULL
    * Approach1: Dummy Node, Time:O(n), Space:O(1)
      * Algo:
        * 1. ind the position before start of reverse node (prev_end)
        * 2. Reverse the nodes in the demand range
        * 3. Connect
      * Python:
        ```python
          def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
            if n - m < 1:
                return head

            prev_end = dummy = ListNode(0)
            dummy.next = head
            # step1: find prev_end
            for _ in range(0, m-1):
                prev_end = prev_end.next

            # step2: reverse the nodes in the range
            prev, cur = prev_end, prev_end.next
            for _ in range(0, n-m+1):
                nxt = cur.next
                cur.next = prev
                prev, cur = cur, nxt

            # connect
            prev_end.next.next = cur
            prev_end.next = prev

            return dummy.next
        ```
  * 025: **Reverse** Nodes in **k-Group** (H)
    * Definition:
      * Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.k is a positive integer and is less than or equal to the length of the linked list.
      * If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
      * Do not need to check k
    * Approach1: Two Passes, Time:O(n), Space:O(1)
      * Python Solution
        ```python
        def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
          if not head or k <= 1:
            return head

          n = 0
          cur = head
          while cur:
              n += 1
              cur = cur.next

          if k > n:
              return head

          # total group number
          g_cnt = n // k
          prev = prev_end = dummy = ListNode(0)
          cur = dummy.next = head
          for _ in range(g_cnt):
              for _ in range(k):
                  nxt = cur.next
                  cur.next = prev
                  prev, cur = cur, nxt

              prev_end_next = prev_end.next
              prev_end_next.next = cur
              prev_end.next = prev

              prev = prev_end = prev_end_next

          return dummy.next
        ```
    * Approach2: One Pass, Time:O(n), Space:O(1)
      * Python Solution
      ```python
      def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head or k <= 1:
            return head

        prev_end = dummy = ListNode(0)
        runner = dummy.next = head
        cnt = 0

        while runner:
            cnt += 1
            runner = runner.next

            if cnt % k == 0:
                # reverse
                prev = prev_end
                cur = prev_end.next
                for _ in range(k):
                    nxt = cur.next
                    cur.next = prev
                    prev, cur = cur, nxt

                # connect
                next_prev_end = prev_end.next
                prev_end.next = prev
                next_prev_end.next = cur
                prev_end = next_prev_end

        return dummy.next
      ```
  * 024: **Swap** Nodes in **Pair** (M) *
    * Approach1: Dummy Node, Time:O(n), Space:O(1)
      * Use 3 pointers, prev ,current and next.
      * Python
        ```python
        def swapPairs(self, head: ListNode) -> ListNode:
          if not head:
              return

          prev = dummy = ListNode(0)
          cur = dummy.next = head

          while cur and cur.next:
              nxt = cur.next
              prev.next = nxt
              cur.next = nxt.next
              nxt.next = cur

              prev, cur = cur, cur.next

          return dummy.next
        ```
  * 143: **Reorder** List (M)
    * Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
    * Example
      * Input: 1->2->3->4->5
      * Output: 1->5->2->4->3.
    * Approach1: Reverse, Time:O(n), Space O(1):
        1. Using the **"Runner"** Techinique to seprate first half and second half of the linked list.
        2. **Reverse the second half** of the linked list.
        3. Combine the first half and second half by iterative through out the second half linked list.
      * Python
        ```python
        def reorderList(self, head: ListNode) -> None:
          """
          Do not return anything, modify head in-place instead.
          """
          if not head or not head.next:
              return

          slow = head
          fast = head.next  # Ensure the 1st part has the same or one more node
          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next

          cur = slow.next
          prev = None
          slow.next = None
          # reverse the second list
          while cur:
              nxt = cur.next
              cur.next = prev
              prev, cur = cur, nxt

          # connect
          first = head
          second = prev
          while second:
              second_nxt = second.next
              second.next = first.next
              first.next = second

              first = second.next
              second = second_nxt
        ```
    * Approach2: Use Stack, Time:O(n), Space O(n):
      * Use a stack to store 2nd part of the linkedlist.
      * Python
        ```python
        def reorderList(self, head: ListNode) -> None:
          # need at least 3 nodes
          if not head or not head.next:
              return

          slow = head
          fast = head.next  # Ensure the 1st part has the same or one more node
          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next

          # push the 2nd part to stack
          cur = slow.next
          slow.next = None
          stack = []
          while cur:
              stack.append(cur)
              cur = cur.next

          # connect
          first = head
          while stack:
              pop = stack.pop()
              pop.next = first.next
              first.next = pop
              first = pop.next
        ```
* **Partition**
  * 061: Rotate list (M)
    * Note
      * The rotate length k may be greater than the length of linked list
    * Approach1: Split and Connect, Time O(n), Space O(1)
      * Algo:
        * old_head -> .. -> new_tail -> new_head -> ... old_tail -> None
        * 1: find the old_tail and length of linkedlist
        * 2: find the new_tail and new_head
        * 3: connect old_tail to old_head
        * 4: connect new_tail to None
        * new_head is the head of the new linked list
      * Python
          ```python
          def rotateRight(self, head: ListNode, k: int) -> ListNode:
            if not head:
                return head
            """
            old_head -> .. -> new_tail -> new_head -> ... old_tail -> None
            """
            n = 1
            old_tail = head
            while old_tail.next:
                length += 1
                old_tail = old_tail.next

            k = k % n
            if k == 0:
                return head

            new_tail = head
            for _ in range(n-k-1):
                new_tail = new_tail.next

            new_head = new_tail.next
            new_tail.next = None
            old_tail.next = head

            return new_head
          ```
  * 086: Partition List (M)
    * Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
    * Approach1: Split and Merge, Time:O(n), Space:O(1)
      * Python
        ```python
        def partition(self, head: ListNode, x: int) -> ListNode:
          if not head:
              return head

          d1 = dummy1 = ListNode(0)
          d2 = dummy2 = ListNode(0)

          cur = head
          while cur:
              if cur.val < x:
                  d1.next = cur
                  d1 = d1.next
              else:
                  d2.next = cur
                  d2 = d2.next
              cur = cur.next

          d1.next = dummy2.next
          d2.next = None

          return dummy1.next
        ```
  * 328: Odd Even Linked List (M)
    * Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.
    * Approach1: Split and Merge. Time:O(n), Space:O(1)
      * Python
        ```python
          def oddEvenList(self, head: ListNode) -> ListNode:
            odd = dummy_odd = ListNode(0)
            even = dummy_even = ListNode(0)
            cur = head

            is_odd = True

            while cur:
                if is_odd:
                    odd.next = cur
                    odd = odd.next
                else:
                    even.next = cur
                    even = even.next

                is_odd = not is_odd
                cur = cur.next

            odd.next = dummy_even.next
            even.next = None

            return dummy_odd.next
      ```
  * 725: Split Linked List in Parts (M)
    * Approach1: Split input list, Time: O(n), Space:O(1)
      * Example:
        * 1.
          * Input:  [1, 2, 3], k = 5
          * Output: [[1], [2], [3], None, None]
        * 2.
          * Input:  [1, 2, 3, 4, 5, 6, 7], k = 5
          * Output: [[1,2], [3,4], [5], [6], [7], None, None]
      * Note:
        * how to determine the partition size ?
      * Python
        ```python
        def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
          partitions = [None] * k
          if not root:
              partitions

          cnt = 0
          cur = root
          while cur:
              cnt += 1
              cur = cur.next

          size, extra = cnt // k, cnt % k
          cur = root
          for p in range(k):
              # run out of the list
              # example: 1->2->3, k = 5
              if not cur:
                  break

              # determine the size of each partition
              partitions[p] = cur
              p_size = size
              if extra > 0:
                  p_size += 1
                  extra -= 1
              for _ in range(p_size-1):
                  cur = cur.next
              cur.next, cur = None, cur.next

          return partitions
        ```
* **Sorting and Merge**
  * 021: Merge Two **Sorted** Lists (E)
    * Approach1: Time:O(n), Space:O(1)
      * Python
        ```python
        def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
          cur = dummy = ListNode(0)
          while l1 and l2:
              if l1.val <= l2.val:
                  cur.next = l1
                  l1 = l1.next
              else:
                  cur.next = l2
                  l2 = l2.next
              cur = cur.next

          if l1:
              cur.next = l1

          if l2:
              cur.next = l2

          return dummy.next
        ```
  * 148: Sort list (M)
      * Ref:
        * https://stackoverflow.com/questions/1525117/whats-the-fastest-algorithm-for-sorting-a-linked-list/1525419#1525419
      * Approach1: Top down merge sort , Time: O(nlog(n), Space:O(logn)
        * Python
          ```python
          def sortList(self, head: ListNode) -> ListNode:
            def merge_2_sorted_lists(l1, l2):
                cur = dummy = ListNode(0)
                while l1 and l2:
                    if l1.val <= l2.val:
                        cur.next = l1
                        l1 = l1.next
                    else:
                        cur.next = l2
                        l2 = l2.next

                    cur = cur.next

                if l1:
                    cur.next = l1

                if l2:
                    cur.next = l2

                return dummy.next

            if not head or not head.next:
                return head

            slow = head
            fast = head.next
            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next

            left = head
            right = slow.next
            slow.next = None  # set tail of left to None
            # left part
            left = self.sortList(head)
            # right part
            right = self.sortList(right)

            head = merge_2_sorted_lists(left, right)

            return head
          ```
      * Approach2: Bottom up merge sort , Time: O(nlog(n), Space:O(1)
        * Algo:
        * Like traditional bottom up sorting algorithm:
          * Start from window size 1 to n-1
          * Scanning from left to right
            * **Split** the linked with window size .
            * **Merge** the splitted linked lists.
          * Note:
            * Need to handle **linking issue** between two sorted lists **after merging**.
          * example:
              ```txt
              window_size = 3
               O->O->O  O->O->O-> O->O->O-> O->O->O->
              prev_end  left      right     next_left
              ```
        * Python
          ```python
          def sortList(self, head: ListNode) -> ListNode:

            def get_len(head):
                cur = head
                n = 0
                while cur:
                    n += 1
                    cur = cur.next
                return n

            def merge_2_sorted_lists(prev_end, l1, l2):
                cur = prev_end

                while l1 and l2:
                    if l1.val <= l2.val:
                        cur.next = l1
                        l1 = l1.next
                    else:
                        cur.next = l2
                        l2 = l2.next

                    cur = cur.next

                if l1:
                    cur.next = l1

                if l2:
                    cur.next = l2

                # move cur to the end
                while cur.next:
                    cur = cur.next

                # cur is tail of the merged list
                return cur

            def split_list_with_k_len(head, k):
                """
                Divide the linked list into two lists,
                while the first list contains first k ndoes
                return the second list's head
                """
                cur = head
                second = None

                for _ in range(k-1):
                    if not cur:
                        break
                    cur = cur.next

                if not cur:
                    return second

                second = cur.next
                cur.next = None

                return second

            if not head or not head.next:
                return head

            dummy = ListNode(0)
            dummy.next = head

            n = get_len(head)
            window_size = 1

            while window_size < n:
                prev_end = dummy
                left = dummy.next

                while left:
                    right = split_list_with_k_len(left, window_size)
                    next_left = split_list_with_k_len(right, window_size)
                    prev_end = merge_2_sorted_lists(prev_end, left, right)
                    left = next_left

                window_size *= 2

            return dummy.next
          ```
  * 023: Merge k **Sorted** Lists (H)
    * Similar Problems:
      * 378:	**Kth Smallest Element** in a Sorted Matrix (M)
    * Assume total n nodes, k lists
    * Approach1: Brute Force: Time:O(nlog(n)), Space:O(sorting)
      * Push and Sort all the nodes
      * Push takes: O(n)
      * Sorintg takes: O(nlog(n))
    * Approach2: **Insertion Sort**, Time:O(nk), Space:O(1)
      * Convert merge k lists problem to merge 2 lists k-1 times.
      * Time Complexity:
        * Assume node for each list is n/k
        * total cost would be (2n+3n+...kn)/k = O(kn)
    * Approach3: **Priority Queue**, (min heap), Time:O(nlog(k)), Space:O(k)
      * Time complexity: O(nlog(k))
        * Generate heap: O(k)
        * Total n operations (n nodes),
          * O(log(k)): put to priority queue
          * O(log(k)): get min from priority queue
      * Space complexity: O(k)
        * quque size
      * Python
        * Implementation1: Use priority Queue
          ```python
          from queue import PriorityQueue

          class Event(object):
              def __init__(self, node):
                  self.node = node

              def __lt__(self, other):
                  return self.node.val < other.node.val

            def mergeKLists(lists: List[ListNode]) -> ListNode:
                cur = dummy = ListNode(0)
                q = PriorityQueue()

                for head in lists:
                    if head:
                        q.put(Event(head))

                # can not use while q here !
                while not q.empty():
                    node = q.get().node
                    cur.next = node
                    cur = cur.next
                    if node.next:
                        q.put(Event(node.next))

                return dummy.next
          ```
        * Implementation2: Use min heap
          ```python
          class Element(object):
            def __init__(self, node):
                self.node = node

            def __lt__(self, other):
              return self.node.val < other.node.val


          def mergeKLists(self, lists: List[ListNode]) -> ListNode:
              # heapify: O(k)
              heap = []
              for head in lists:
                  if head:
                      heap.append(Element(head))
              heapq.heapify(heap)

              cur = dummy = ListNode(0)
              # O(nlogk)
              while heap:
                  pop_min = heap[0].node
                  cur.next = pop_min
                  cur = cur.next
                  if pop_min.next:
                      heapq.heapreplace(heap, Element(pop_min.next))
                  else:
                      heapq.heappop(heap)

              return dummy.next
          ```
    * Approach4: **bottom up Merge Sort**, Time:O(nlog(k)), Space:O(1)
      * Time complexity: O(nlog(k))
        * Total log(k) round:
          * each round need to traverse every nodes
      * Space complexity: O(1)
      * Python
        ```python
        def merge_2_sorted_Lists(self, l1, l2):
          cur = dummy = ListNode(0)
          while l1 and l2:
              if l1.val < l2.val:
                  cur.next = l1
                  l1 = l1.next
              else:
                  cur.next = l2
                  l2 = l2.next
              cur = cur.next

          if l1:
              cur.next = l1

          if l2:
              cur.next = l2

          return dummy.next

        def mergeKLists(self, lists: List[ListNode]) -> ListNode:
            if not lists:
                return None

            # bottom up merge lists
            interval = 1
            while interval < len(lists):
                left = 0
                while left + interval < len(lists):
                    right = left+interval
                    lists[left] = self.merge_2_sorted_Lists(lists[left], lists[left+right])
                    left += (2 * interval)
                interval *= 2

            return lists[0]
        ```
  * 147: Insertion Sort List (M)
    * Approach1: standard insertion sort, Time:O(n^2), Space:O(1)
      * create a new dummy to store sorted list
      * handle the insert val properly
      * Python
      ```python
        def insertionSortList(self, head: ListNode) -> ListNode:

          if not head or not head.next:
              return head

          # new head to store sored list
          prev = dummy = ListNode(0)
          cur = head
          while cur:

              nxt = cur.next
              # Should we move the "prev" back to the dummy (head of the sorted list)
              if prev.val > cur.val:
                  prev = dummy

              # find the correct position
              while prev.next and prev.next.val < cur.val:
                  prev = prev.next


              cur.next = prev.next
              prev.next = cur

              cur = nxt


          return dummy.next
      ```
* **Add numbers**:
  * 002: Add Two Numbers (M)
    * Example:
      * From left to right
      * Input:
        * (2 -> 4 -> 3) + (5 -> 6 -> 4)
      * Output:
        * 7 -> 0 -> 8
    * Approach1: one pass, Time:O(n), Space:O(1)
      * Don't forget the **last carry**.
      * Python
        ```python
        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
          cur = dummy = ListNode(0)
          carry = 0

          while l1 or l2 or carry:
              val = carry

              if l1:
                  val += l1.val
                  l1 = l1.next
              if l2:
                  val += l2.val
                  l2 = l2.next

              cur.next = ListNode(val % 10)
              carry = val // 10
              cur = cur.next

          return dummy.next
        ```
  * 445: Add Two Numbers II (M)
    * Example:
      * From right to left
      * Input:
        * (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
      * Output:
        * 7 -> 8 -> 0 -> 7
    * Approach1: Reverse and Add, Time:O(n), Space:O(1)
      * Python
        ```python
        def reverse(self, head):
            prev = None
            cur = head
            while cur:
                nxt = cur.next
                cur.next = prev
                prev, cur = cur, nxt

            return prev

        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
            l1 = self.reverse(l1)
            l2 = self.reverse(l2)
            head = None
            carry = 0

            while l1 or l2 or carry:
                val = carry
                if l1:
                    val += l1.val
                    l1 = l1.next

                if l2:
                    val += l2.val
                    l2 = l2.next

                 new = ListNode(val%10)
                 new.next = head
                 head = new
                 carry = val // 10

            return self.reverse(dummy.next)
        ```
    * Approach2: Use Stack, Time:O(n), Space:O(1)
      * Python
        ```python
        def push_to_stack(self, head):
          stack = list()
          cur = head
          while cur:
              stack.append(cur)
              cur = cur.next

          return stack

        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
          s1 = self.push_to_stack(l1)
          s2 = self.push_to_stack(l2)

          head = None
          carry = 0
          while s1 or s2 or carry:
              val = carry
              if s1:
                  val += s1.pop().val

              if s2:
                  val += s2.pop().val

              new = ListNode(val%10)
              new.next = head
              head = new
              carry = val // 10

          return head
        ```
  * 369: Plus One Linked List (M)
    * Example:
      * 1->2->3  ==> 1->2->4
      * 1->9->9  ==> 2->0->0
      * 9->9->9  ==> 1->0->0->0
    * Approach1: Reverse, Time:O(n), Space:O(1)
      * Algo
        * 1. **Reverse** given linked list. For example, 1-> 9-> 9 -> 9 is converted to 9-> 9 -> 9 ->1.
        * 2. Start traversing linked list from leftmost node and add 1 to it. If there is a carry, move to the next node. Keep moving to the next node while there is a carry.
        * 3. **Reverse** modified linked list and return head.
      * Python Solution
        ```python
        def plusOne(self, head: ListNode) -> ListNode:
          def reverse(cur):
              prev = None
              while cur:
                  nxt = cur.next
                  cur.next = prev
                  prev, cur = cur, nxt

              return prev

          if not head:
              return head

          # 1. reverse
          head = reverse(head)

          # 2. plus one
          cur = head
          carry = 1
          while cur:
              val = carry + cur.val
              cur.val = val % 10
              carry = val // 10

              # create one
              if carry and not cur.next:
                  cur.next = ListNode(carry)
                  break

              cur = cur.next

          # 3. reverse back
          head = reverse(head)

          return head
        ```
* Others
  * 160: Intersection of Two Linked Lists (E)
    * Approach1: Use **difference** of length
      * Python Solution
        ```python
          def getIntersectionNode(self, headA, headB):

            def get_length(runner):
                n = 0
                while runner:
                    n += 1
                    runner = runner.next
                return n

            len_a , len_b = get_length(headA), get_length(headB)

            diff = len_a - len_b
            if diff >= 0:
                long_l, short_l = headA, headB
            elif diff < 0:
                long_l, short_l = headB, headA

            # don't forget to use
            for _ in range(abs(diff)):
                long_l = long_l.next

            res = None
            while short_l:
                if short_l is long_l:
                    res = short_l
                    break

                short_l = short_l.next
                long_l = long_l.next

            return res
        ```
  * 234: **Palindrome** Linked List (M)
    * Approach1: Reverse, Time:O(n), Space: O(1):
      * Reverse first half of the linked list, but it is not a pratical solution since we should not modify the constant function of the input.
      * Python
        ```python
        def isPalindrome(self, head: ListNode) -> bool:
          # node <= 1
          if not head or not head.next:
              return True

          slow = head
          fast = head.next # Ensure the 1st part has the same or one more node
          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next

          cur = slow.next
          prev = slow.next = None

          # reverse second part
          while cur:
              nxt = cur.next
              cur.next = prev
              prev, cur = cur, nxt

          is_palindrom = True
          first = head
          second = prev

          # compare
          while second:
              if second.val != first.val:
                  is_palindrom = False
                  break
              first = first.next
              second = second.next

          return is_palindrom
        ```
    * Approach2: Use Stack, Time:O(n), Space: O(n):
      * Python
        ```python
        def isPalindrome(self, head: ListNode) -> bool:
          if not head or not head.next:
              return True

          """
          Ensure the 1st part has the same or one more node
          """
          slow, fast = head, head.next
          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next

          stack = list()
          cur = slow.next
          while cur:
              stack.append(cur)
              cur = cur.next

          is_palindrom = True
          cur = head
          while stack:
              if cur.val != stack.pop().val:
                  is_palindrom = False
                  break
              cur = cur.next

          return is_palindrom
        ```
* TODO
  * 1171: Remove Zero Sum Consecutive Nodes from Linked List
### Stack and Queue
  * 155: Min Stack (E)
    * Space (n)
      * Use extra space to keep minimum
      * Python Solution
      ```python
      class MinStack:
        def __init__(self):
            """
            initialize your data structure here.
            """
            self.stack = list()
            self.mins = list()

        def push(self, x: int) -> None:
            self.stack.append(x)
            if not self.mins:
                self.mins.append(x)
            else:
                self.mins.append(min(self.mins[-1], x))


        def pop(self) -> None:
            if not self.stack:
                return

            self.stack.pop()
            self.mins.pop()

        def top(self) -> int:
            return self.stack[-1]

        def getMin(self) -> int:
            return self.mins[-1]
      ```
  * 232: Implement Queue using Stacks (E)
    * Approach1: Use 2 stacks, push:O(1), pop: average, O(1)
      * Ref:
        * https://leetcode.com/articles/implement-queue-using-stacks/
      * Python
        ```python
        class MyQueue:
          def __init__(self):
              """
              Initialize your data structure here.
              """
              self.push_stack = []
              self.pop_stack = []

          def push(self, x: int) -> None:
              """
              Push element x to the back of queue.
              """
              self.push_stack.append(x)

          def pop(self) -> int:
              """
              Removes the element from in front of queue and returns that element.
              """
              if self.empty():
                  return None

              return self.pop_stack.pop()

          def peek(self) -> int:
              """
              Get the front element.
              """
              if self.empty():
                  return None

              return self.pop_stack[-1]

          def empty(self) -> bool:
              """
              Returns whether the queue is empty.
              """
              if len(self.pop_stack) > 0:
                  return False

              while self.push_stack:
                  self.pop_stack.append(self.push_stack.pop())

              return len(self.pop_stack) == 0
        ```
  * 225: Implement Stack using Queues (E)
    * Approach1: 1 queues, Push:O(1), Pop:O(n)
      * Python
        ```python
        class MyStack:
          def __init__(self):
              """
              Initialize your data structure here.
              """
              self.queue = collections.deque()

          def push(self, x: int) -> None:
              """
              Push element x onto stack.
              """
              self.queue.append(x)

          def pop(self) -> int:
              """
              Removes the element on top of the stack and returns that element.
              """
              q_len = len(self.queue)
              for _ in range(q_len-1):
                  self.queue.append(self.queue.popleft())

              return self.queue.popleft()

          def top(self) -> int:
              """
              Get the top element.
              """
              return self.queue[-1]

          def empty(self) -> bool:
              """
              Returns whether the stack is empty.
              """
              return not len(self.queue)
        ```
    * Approach3: 1 queues, Push:O(n), Pop:O(1)
      * Python
        ```python
        class MyStack:
          def __init__(self):
              """
              Initialize your data structure here.
              """
              self.queue = collections.deque()

          def push(self, x: int) -> None:
              """
              Push element x onto stack.
              """
              self.queue.append(x)
              q_len = len(self.queue)
              for _ in range(q_len-1):
                  self.queue.append(self.queue.popleft())

          def pop(self) -> int:
              """
              Removes the element on top of the stack and returns that element.
              """
              return self.queue.popleft()

          def top(self) -> int:
              """
              Get the top element.
              """
              return self.queue[0]

          def empty(self) -> bool:
              """
              Returns whether the stack is empty.
              """
              return not len(self.queue)
        ```
  * 394: Decode String (M)
    * Approach1: Use 2 stacks: Time:O(kn), Space:O(kn)
      * Ref:
        * https://leetcode.com/problems/decode-string/discuss/87534/Simple-Java-Solution-using-Stack
      * Python
        ```python
        def decodeString(self, s: str) -> str:
          buf = []
          cnt_s = []
          decode_s = []
          ord_0 = ord('0')

          i = 0
          while i < len(s):
              c = s[i]
              if c.isdigit():
                  cnt = 0
                  while s[i].isdigit():
                      cnt = cnt * 10 + (ord(s[i]) - ord_0)
                      i+= 1
                  cnt_s.append(cnt)

              elif c == '[':
                  decode_s.append(buf)
                  buf = []
                  i += 1

              elif c == ']':
                  repeat_cnt = cnt_s.pop()
                  repeat_str = "".join(buf)
                  buf = decode_s.pop()
                  for _ in range(repeat_cnt):
                      buf.append(repeat_str)
                  i += 1

              else:
                  buf.append(c)
                  i += 1

            return "".join(buf)
          ```
  * 255: Verify Preorder Sequence in Binary Search Tree (M)
    * Ref:
      * https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/discuss/68185/C%2B%2B-easy-to-understand-solution-with-thought-process-and-detailed-explanation
    * The key is how to find the lower bound
    * Preorder sequence: root[left_subtree][right_subtree]
      * Since left_subtree_val < root < right_subtree_val, when we find the first val which is greater than the root, it means that root should be the current lower bound
      * stack would be (r1, r2, r3) pop until find the correct root node
    * Approach1: Stack
      * Python
        ```python
        def verifyPreorder(self, preorder: List[int]) -> bool:
          lower = float('-inf')
          stack = []
          is_bst = True
          for n in preorder:
              if n < lower:
                  is_bst = False
                  break

              while stack and n > stack[-1]:
                  # pop until find the current lower bound
                  lower = stack.pop()

              stack.append(n)

          return is_bst
        ```
  * 341: Flatten Nested List Iterator (M)
    * Approach1: Use stack
      * Python
        ```python
        class NestedIterator(object):
          def __init__(self, nestedList):
              self.stack = nestedList[::-1]

          def next(self):
              return self.stack.pop().getInteger()

          def hasNext(self):
              while self.stack:
                  top = self.stack[-1]
                  if top.isInteger():
                      return True
                  else:
                      self.stack = self.stack[:-1] + top.getList()[::-1]

              return False
        ```
    * Approach2: Use deque
      * Python
      ```python
      class NestedIterator(object):

          def __init__(self, nestedList):
              self.dq = collections.deque()
              self.dq.extend(nestedList)

          def next(self):
              return self.dq.popleft().getInteger()

          def hasNext(self):
              while self.dq:
                  top = self.dq[0]
                  if top.isInteger():
                      return True
                  else:
                      self.dq.popleft()
                      self.dq.extendleft(top.getList()[::-1])

              return False
      ```
### Heap (Priority Queue)
  * Note:
    * For Python, heapq is min-heap
  * 023: Merge k Sorted Lists (H) (LinkedList)
    * Ｐleaser refer linked list section (use min heap)
  * 253: Meeting Rooms II (M) (Array)
    * Please refer array section, use min heap
  * 703. **Kth Largest** Element in a Stream (E)
    * Approach1: min heap, Time: O(log(k)), Space: O(k)
      * Time: O(log(k))
        * __init__: O(nlog(k))
          * n items for heap push
        * add:
          * heapreplace and heappush take O(log(k))
      * Space: O(k)
        * Capacity of heap size
      * Python Solution:
        ```python
        class KthLargest:

          def __init__(self, k: int, nums: List[int]):
              # You may assume that nums' length ≥ k-1 and k ≥ 1.
              self.heap = list()
              self.cap = k
              for num in nums:
                  self._add(num)

          def _add(self, val):
              if len(self.heap) < self.cap:
                  heapq.heappush(self.heap, val)
              else:
                  # replace the minimize element in the heap
                  if val > self.heap[0]:
                      heapq.heapreplace(self.heap, val)

          def add(self, val: int) -> int:
              self._add(val)
              return self.heap[0]
        ```
  * 215: **Kth Largest** Element in an Array (M)
    * Find the kth largest element in an unsorted array.
    * Approach1: Sorting, Time: O(nlog(n)), Space:O(log(n))
    * Approach2: min heap, Time: O(nlog(k)), Space: O(k)
      * keep the k element in the minimum heap
      * Time Complexity: O(nlog(l))
        * **heappush**
          * Append to the tail of list, and make bubble up comparison cost log(k)
        * **heapreplace**
          * Replace the min, and make bubble down operation, cost log(k)
      * Python Solution
        ```python
        def findKthLargest(self, nums: List[int], k: int) -> int:
          heap = []
          for num in nums:
              if len(heap) < k:
                  heapq.heappush(heap, num)
              elif num > heap[0]:
                  # replace the minimize element in the heap
                  heapq.heapreplace(heap, num)

          return heap[0]
        ```
    * Approach3: Quick select, Time: O(n), Space:O(1)
      * Python Solution:
        ```python
        class Solution:
          def partition(self, nums, start, end):
              # to avoid worst case
              pivot_ran = random.randint(start, end)

              pivot = end
              nums[pivot_ran], nums[pivot] = nums[pivot], nums[pivot_ran]
              pivot_val = nums[pivot]

              border = start
              for cur in range(start, end):
                  if nums[cur] <= pivot_val:
                      nums[border], nums[cur] = nums[cur], nums[border]
                      border += 1

              nums[border], nums[pivot] = nums[pivot], nums[border]
              return border

          def quick_select(self, nums, start, end, k_smallest):
              while start <= end:
                pivot = self.partition(nums, start, end)
                if k_smallest == pivot:
                    return nums[pivot]
                elif k_smallest < pivot:
                    end = pivot - 1
                else:
                    start = pivot + 1

          def findKthLargest(self, nums: List[int], k: int) -> int:
              if k > len(nums):
                  return None

              # k_smallest is len(nums)-k
              return self.quick_select(nums, 0, len(nums)-1, len(nums)-k)
        ```
  * 347: **Top K Frequent** Elements (M)
    * Given a non-empty array of integers, return the **k most frequent elements**.
    * If k == 1, Hash Table, Time:O(n), Space:O(n)
      * Use dictionary and keep the max key
    * If k > 1, Hash Table + min heap Time:O(nlog(k)), Space:O(n)
      * Time O(nlog(k))
        * build hash map :O(n)
        * build heap :O(nlog(k))
      * Space: O(n)
        * O(n) for hash table
        * O(k) for heap
      * Approach1: heapq-1:
        ```python
        class Element(object):
          def __init__(self, key, count):
              self.key = key
              self.count = count

          def __lt__(self, other):
              return self.count < other.count

          def __repr__(self):
              return str(self.key)

        def topKFrequent(nums: List[int], k: int) -> List[int]:
            d = collections.Counter(nums)
            heap = list()

            for key, cnt in d.items():
                if len(heap) < k:
                    heapq.heappush(heap, Element(key, cnt))
                else:
                    e = Element(key, cnt)
                    if heap[0] < e:
                        heapq.heapreplace(heap, e)

            return [e.key for e in heap]
        ```
      * Approach2: heapq-2
        ```python
        def topKFrequent(self, nums: List[int], k: int) -> List[int]:
          counter = collections.Counter(nums)
          return heapq.nlargest(k, counter.keys(), key=counter.get)
        ```
  * 692: **Top K Frequent** Words (M)
    * Approach1: hash table + min heap, Time:O(nlogk), Space:O(n):
      * Time:
        * Create Hash Table: O(n)
        * Insert to Heap: O(nlogk)
        * Pop from heap: O(klogk)
      * Space:
        * Hash Table: O(n)
        * Heap: O(k)
      * Python Solution:
        ```python
        class Element(object):

          def __init__(self, key, count):
              self.key = key
              self.count = count

          def __lt__(self, other):
              if self.count == other.count:
                  # If two words have the same frequency,
                  # then the word with the lower alphabetical order has higher priority
                  return self.key > other.key
              else:
                  return self.count < other.count

          def __repr__(self):
              return str(f'{self.key}:{self.count}')

        def topKFrequent(self, words: List[str], k: int) -> List[str]:
            d = collections.Counter(words)
            heap = list()

            for key, cnt in d.items():
                if len(heap) < k:
                    heapq.heappush(heap, Element(key, cnt))
                else:
                    e = Element(key, cnt)
                    if heap[0] < e:
                        heapq.heapreplace(heap, e)

            ret = list()
            while heap:
                ret.append(heapq.heappop(heap).key)
            return ret[::-1]
        ```
  * 373: **Find K Pairs** with Smallest Sums (M)
    * Approach1: Brute Force, Time:O(nm), Space:O(k)
      * Python
        ```python
        class Element(object):
          def __init__(self, pair):
              self.pair = pair

          # for max heap
          def __lt__(self, other):
              return (self.pair[0] + self.pair[1]) > (other.pair[0] + other.pair[1])

          def __repr__(self):
              return str(self.pair)

        def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
            heap = []

            for n1 in nums1:
                for n2 in nums2:
                    n_pair = (n1, n2)
                    if len(heap) < k:
                        heapq.heappush(heap, Element(n_pair))
                        continue

                    cur_max_pair = heap[0].pair
                    if (n_pair[0] + n_pair[1]) < (cur_max_pair[0] + cur_max_pair[1]):
                        heapq.heapreplace(heap, Element(n_pair))
                    else:
                        break

            return [e.pair for e in heap]
        ```
    * Approach2: Use Matrix to simulate merge k sorted list, Time:O(klogn), Space:O(min(k,n))
      * Example:
        * nums1 = [1,3,5]
        * nums2 = [2,4,6]
        * list1: (1,2) -> (1,4) -> (1,6)
        * list2: (3,2) -> (3,4) -> (3,6)
        * list3: (5,2) -> (5,4) -> (5,6)
      * Python:
        ```python
        class Element(object):
          def __init__(self, r, c, val):
              self.r = r
              self.c = c
              self.val = val

          def __lt__(self, other):
              return self.val < other.val

          def __repr__(self):
              return str(self.val)

        def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
          if not nums1 or not nums2:
              return []

          heap = []
          res = []

          # Use matrix to simulate merge k sorted list
          # push the heads of the lists
          # we don't need to push more than k elements
          for r in range(min(k, len(nums1))):
            heap.append(Element(r, 0, nums1[r] + nums2[0]))
          heapq.heapify(heap)

          while heap and len(res) < k:
              r, c = heap[0].r, heap[0].c
              if c + 1 < len(nums2):
                  heapq.heapreplace(heap, Element(r, c+1, nums1[r]+nums2[c+1]))
              else:
                  heapq.heappop(heap)

              res.append((nums1[r], nums2[c]))

          return res
        ```
  * 295: Find Median from Data Stream (H)
    * Approach1: Insertion Sort, Time:O(n), Space:O(n)
      * Time: O(n)
        * add num: O(n)
          * search: O(logn), binary search to find correct position takes
          * insertion: O(n), since elements have to be shifted inside the container
        * find median: O(1)
      * Python
        ```python
        class MedianFinder:
          def __init__(self):
              self.nums = []

          def addNum(self, num: int) -> None:
              self.nums.append(num)
              # from insert-1 to 0
              for i in range(len(self.nums)-2, -1, -1):
                  if self.nums[i] > self.nums[i+1]:
                      self.nums[i], self.nums[i+1] = self.nums[i+1], self.nums[i]
                  else:
                      break

          def findMedian(self) -> float:
              n = len(self.nums)

              if n == 0:
                  return None

              if n % 2: # odd
                  return self.nums[n//2]
              else: # even
                  return (self.nums[n//2] + self.nums[n//2-1]) / 2
              ```
    * Approach2: 2 heaps, Time:O(n), Space:O(n)
      * two heaps:
        * max heap (lo): A max-heap lo to store the smaller half of the numbers
        * min heap (hi): A min-heap hi to store the larger half of the numbers
        * This leads us to a huge point of pain in this approach: **balancing the two heaps**!
      * The concept is similar to median of 2 sorted array
      * lo (max heap) is allowed to hold n + 1 elements, while hi can hold n elements
        * example:
          * for 2n element, lo holds n, hi hold n as well.
          * for 2n+1 elements, lo hold n+1, hi hold n.
      * Python
        ```python
        class MedianFinder:
          def __init__(self):
              self.lo = []  # max-heap, keep the smaller half of the numbers
              self.hi = []  # min-heap, keep the larger half of the number

          def addNum(self, num: int) -> None:
              # push to lo
              if len(self.lo) == len(self.hi):
                  pop = heapq.heappushpop(self.hi, num)
                  # revert the number to transfer min-heap to max-heap
                  heapq.heappush(self.lo, -pop)

              # push to hi
              else: # len(self.lo) > len(self.hi)
                  # revert the number to transfer min-heap to max-heap
                  pop = heapq.heappushpop(self.lo, -num)
                  heapq.heappush(self.hi, -pop)

          def findMedian(self) -> float:

              if len(self.lo) == len(self.hi):
                  return  (-self.lo[0] + self.hi[0]) / 2
              else:
                  return -self.lo[0]
        ```
  * 313: Super Ugly Numbe (M)
  * 218: The Skyline Problem (H)
### Cache
  * 146: LRU Cache (M)
    * Note:
      * Don't forget to update val of key in put operation.
    * Approach1: ordered dict
    * Approach2: dict + doubly linked list
      * Put
        * new record
          * insert_head
          * pop_tail if necessary
        * already existing
          * move_to_head
      * Get
        * move_to_head
      * For Doubly linked list
        * Use dummy nodes for head and tail
      * Python
        ```python
        class DLinkedNode(object):
            def __init__(self, key=None, val=None):
                # key is necessary for key pop operation
                self.key = key
                self.val = val
                self.prev = None
                self.next = None


        class DLinkedList(object):

            def __init__(self):
                self.head = DLinkedNode(0)
                self.tail = DLinkedNode(0)

                self.head.next = self.tail
                self.tail.prev = self.head

            def insert_to_head(self, node) -> None:
                node.prev = self.head
                node.next = self.head.next

                self.head.next.prev = node
                self.head.next = node

            def remove_node(self, node) -> None:
                node.prev.next = node.next
                node.next.prev = node.prev

            def pop_tail(self) -> DLinkedNode:
                if self.tail.prev is self.head:
                    return None

                pop = self.tail.prev
                self.remove_node(pop)
                return pop

            def move_to_head(self, node) -> None:
                self.remove_node(node)
                self.insert_to_head(node)


        class LRUCache:
            def __init__(self, capacity: int):
                self.nodes = dict()
                self.list = DLinkedList()
                self.cap = capacity
                self.len = 0

            def get(self, key: int) -> int:
                if key not in self.nodes:
                    return -1

                node = self.nodes[key]
                self.list.move_to_head(node)
                return node.val

            def put(self, key: int, value: int) -> None:
                if key in self.nodes:
                    node = self.nodes[key]
                    node.val = value
                    self.list.move_to_head(node)
                else:
                    if self.len == self.cap:
                        pop = self.list.pop_tail()
                        self.nodes.pop(pop.key)
                        self.len -= 1

                    new = DLinkedNode(key, value)
                    self.nodes[key] = new
                    self.list.insert_to_head(new)
                    self.len += 1
        ```
  * 460: LFU Cache (H)
    * Example:
      ```txt
      LFUCache cache = new LFUCache( 2 /* capacity */ );
      cache.put(1, 1);
      cache.put(2, 2);
      cache.get(1);       // returns 1
      cache.put(3, 3);    // evicts key 2
      cache.get(2);       // returns -1 (not found)
      cache.get(3);       // returns 3.
      cache.put(4, 4);    // evicts key 1.
      cache.get(1);       // returns -1 (not found)
      cache.get(3);       // returns 3
      cache.get(4);       // returns 4
      ```
    * Note:
      * How to keep the sequence ?? (frequency)
    * Approach1: 2 dictionaries + Doubly LinkedList
      * Ref:
        * https://leetcode.com/problems/lfu-cache/discuss/207673/Python-concise-solution-**detailed**-explanation%3A-Two-dict-%2B-Doubly-linked-list
      * Data Structures
        * **self.nodes**
          * Each key is mapping to the corresponding node, where we can retrieve the node in O(1) time.
        * **self.freq_lists**
          * Each frequency freq is mapped to a Doubly Linked List (), where all nodes in the DLinkedList have the same frequency, freq. Moreover, each node will be always inserted in the head (indicating most recently used).
        * **self.min_freq**
          * A minimum frequency is maintained to keep track of the minimum frequency of across all nodes in this cache, such that the DLinkedList with the min frequency can always be retrieved in O(1) time.
      * Operations:
        * get:
          * query the node by calling self._node[key]
          * find the frequency by checking node.freq, assigned as f, and query the DLinkedList that this node is in, through calling self._freq[f]
          * pop this node
          * update node's frequence, append the node to the new DLinkedList with frequency f+1
          * if the DLinkedList is empty and self._minfreq == f, update self._minfreq to f+1.
          * return node.val
        * put:
          * If key is already in cache, do the same thing as get(key), and update node.val as value
          * Otherwise:
            * if the cache is full, pop the least frequenly used element (*)
            * add new node to self._node
            * add new node to self._freq[1]
            * reset self._minfreq to 1
      * Python
        ```python
        class DLinkedNode(object):
          def __init__(self, key=None, val=None, freq=0):
              # key is necessary for key pop operation
              self.key = key
              self.val = val
              self.freq = freq
              self.prev = None
              self.next = None


        class DLinkedList(object):

            def __init__(self):
                self.head = DLinkedNode(0)
                self.tail = DLinkedNode(0)
                self.head.next = self.tail
                self.tail.prev = self.head
                self.len = 0

            def __len__(self) -> int:
                return self.len

            def insert_to_head(self, node) -> None:
                node.prev = self.head
                node.next = self.head.next

                self.head.next.prev = node
                self.head.next = node
                self.len += 1

            def remove_node(self, node) -> None:
                node.prev.next = node.next
                node.next.prev = node.prev
                self.len -= 1

            def pop_tail(self) -> DLinkedNode:
                if not self.len:
                    return None

                pop = self.tail.prev
                self.remove_node(pop)
                return pop


        class LFUCache:

            def __init__(self, capacity: int):
                self.nodes = dict()
                self.freq_lists = collections.defaultdict(DLinkedList)
                self.min_freq = 0
                self.cap = capacity
                self.len = 0

            def _update(self, node) -> None:
                # 1. remove node from old linked list
                prev_list = self.freq_lists[node.freq]
                prev_list.remove_node(node)

                # 2. update min_freq if lengh of prev_list is 0
                if self.min_freq == node.freq and not prev_list:
                    self.min_freq += 1

                # 3. increase freq and append the node to the new list
                node.freq += 1
                self.freq_lists[node.freq].insert_to_head(node)

            def get(self, key: int) -> int:
                if key not in self.nodes:
                    return -1

                node = self.nodes[key]
                self._update(node)
                return node.val

            def put(self, key: int, value: int) -> None:

                if self.cap <= 0:
                    return

                if key in self.nodes:
                    node = self.nodes[key]
                    self._update(node)
                    node.val = value

                else:
                    # 1. pop node from linked list with min frequency if necessary
                    if self.len == self.cap:
                        pop = self.freq_lists[self.min_freq].pop_tail()
                        self.nodes.pop(pop.key)
                        self.len -= 1

                    # 2. add new node
                    freq = 1
                    new = DLinkedNode(key, value, freq)
                    self.nodes[key] = new
                    pop = self.freq_lists[freq].insert_to_head(new)

                    # 3. update freq and len
                    self.min_freq = freq
                    self.len += 1
        ```
### Tree
  * **Preorder**
    * 144: Binary Tree Preorder **Traversal** (M)
      * Approach1: Recursive, Time:O(n), Space:O(n):
        * Python Solution:
          ```python
          def preorderTraversal(self, root: TreeNode) -> List[int]:
            def _preorder(node):
                if not node:
                    return

                output.append(node.val)
                _preorder(node.left)
                _preorder(node.right)

            output = []
            _preorder(root)
            return output
          ```
      * Approach2: Iterative1, Time:O(n), Space:O(n):
        * Python Solution
          ```python
          def preorderTraversal(self, root: TreeNode) -> List[int]:
              if not root:
                  return []

              visits = list()
              stack = list()
              stack.append(root)

              while stack:
                  node = stack.pop()
                  visits.append(node.val)

                  if node.right:
                      stack.append(node.right)

                  if node.left:
                      stack.append(node.left)

              return visits
          ```
      * Approach3: Iterative2, Time:O(n), Space:O(n):
        * Python Solution
          ```python
          def preorderTraversal(self, root: TreeNode) -> List[int]:
            if not root:
                return []

            stack = []
            output = []

            cur = root
            while cur or stack:
                if not cur:
                    cur = stack.pop()

                output.append(cur.val)
                if cur.right:
                    stack.append(cur.right)
                cur = cur.left

            return output
          ```
    * 226. **Invert** Binary Tree (E)
      * Approach1: DFS, Recursive, Time:O(n), Space:O(n)
          ```python
          def invertTree(self, root: TreeNode) -> TreeNode:
            def _invert(node):

                if not node:
                    return

                node.left, node.right = node.right, node.left
                _invert(node.left)
                _invert(node.right)

            _invert(root)
            return root
          ```
      * Approach2: BFS, Iterative, Time:O(n), Space:O(n)
          ```python
          def invertTree(self, root: TreeNode) -> TreeNode:
            if not root:
                return root

            q = collections.deque([root])
            while q:
                node = q.popleft()
                node.left, node.right = node.right, node.left
                if node.left:
                    q.append(node.left)

                if node.right:
                    q.append(node.right)

            return root
          ```
    * 100: **Same** Tree (E)
      * Approach1: BFS Time:O(n), Space:O(n)
          * Python
          ```python
          def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
            q = collections.deque([(p, q)])
            res = True

            while q:
                node_p, node_q = q.popleft()
                # 0 node
                if not node_p and not node_q:
                    continue
                # 1 node
                if not node_p or not node_q:
                    res = False
                    break

                # 2 nodes
                if node_p.val != node_q.val:
                    res = False
                    break

                q.append((node_p.left, node_q.left))
                q.append((node_p.right, node_q.right))

            return res
          ```
      * Approach2: DFS Time:O(n), Space:O(n)
        * Python
          ```python
          def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
            def is_same_tree(node_p, node_q):
                # 0 node
                if not node_p and not node_q:
                    return True

                # 1 node
                if not node_p or not node_q:
                    return False

                if node_p.val != node_q.val:
                    return False

                # left tree
                if not is_same_tree(node_p.left, node_q.left):
                    return False

                # right tree
                if not is_same_tree(node_p.right, node_q.right):
                    return False

                return True

            return is_same_tree(p, q)
          ```
    * 111: **Minimum Depth** of Binary Tree	(E)
      * Notice the definiton of leaf node (no children)
      * So root can be the leaf node if it does not have children.
      * Approach1: DFS Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def minDepth(self, root: TreeNode) -> int:
            def dfs(node, depth):
                nonlocal min_depth
                if not node.left and not node.right:
                    min_depth = min(min_depth, depth)
                    return

                if node.left:
                    dfs(node.left, depth+1)

                if node.right:
                    dfs(node.right, depth+1)

            if not root:
                return 0

            min_depth = float('inf')
            dfs(root, 1)
            return min_depth
          ```
      * Approach2: DFS Recursive, Bottom up level cnt, postorder
        * Python
          ```python
          def minDepth(self, root: TreeNode) -> int:
            def dfs(node):
                if not node.left and not node.right:
                    return 1

                min_depth = float('inf')

                if node.left:
                    min_depth = min(min_depth, dfs(node.left))

                if node.right:
                    min_depth = min(min_depth, dfs(node.right))

                return min_depth + 1

            if not root:
                return 0

            return dfs(root)
          ```
      * Approach3: BFS Iterative, Top Down, Time:O(n), Space:O(n)
        * Python
          ```python
          def minDepth(self, root: TreeNode) -> int:
            if not root:
                return 0

            # node and depth
            q = collections.deque([(root, 1)])
            min_depth = float('inf')

            while q:
                node, depth = q.popleft()
                # this is a leaf node
                if not node.left and not node.right:
                    min_depth = min(min_depth, depth)
                else:
                  if node.left:
                      q.append((node.left, depth+1))

                  if node.right:
                      q.append((node.right, depth+1))

            return min_depth
          ```
    * 104: **Maximum Depth** of Binary Tree (E)
      * Approach1: DFS Recursive, top down level cnt Time:O(n), Space:O(n)
        * Python
          ```python
          def maxDepth(self, root: TreeNode) -> int:
            def dfs(node, depth):
                nonlocal max_depth

                if not node.left and not node.right:
                    max_depth = max(max_depth, depth)

                if node.left:
                    dfs(node.left, depth+1)

                if node.right:
                    dfs(node.right, depth+1)

            if not root:
                return 0

            max_depth = 0
            dfs(root, 1)
            return max_depth
          ```
      * Approach2: DFS Recursive, Bottom up level cnt, postorder
        * Python
          ```python
          def maxDepth(self, root: TreeNode) -> int:
            def dfs(node):
                if not node.left and not node.right:
                    return 1

                max_depth = 0
                if node.left:
                    max_depth = max(max_depth, dfs(node.left))

                if node.right:
                    max_depth = max(max_depth, dfs(node.right))

                return max_depth + 1

            if not root:
                return 0

            return dfs(root)
          ```
      * Approach3: BFS Iterative, top down level cnt
        * Python
        ```python
        def maxDepth(self, root: TreeNode) -> int:
          if not root:
              return 0

          max_depth = -float('inf')
          q = collections.deque([(root, 1)])

          while q:
              node, depth = q.popleft()

              if not node.left and not node.right:
                  max_depth = max(max_depth, depth)

              else:
                  if node.left:
                      q.append((node.left, depth+1))

                  if node.right:
                      q.append((node.right, depth+1))

          return max_depth
        ```
    * 298: Binary Tree **Longest Consecutive** Sequence (M)
      * Approach1-1: BFS Iterative, from the perspective of cur, Time:O(n), Space:O(n)
        * Python
          ```python
          def longestConsecutive(self, root: TreeNode) -> int:
            if not root:
                return 0

            dummy_val = root.val # ensue there is no consecutive for root
            q = collections.deque([(root, dummy_val, 0)])
            max_path = 0
            while q:
                # current node, prev val, prev acc path length
                node, p_val, p_acc_path = q.popleft()

                if node.val == p_val+1:
                    acc_path = p_acc_path + 1
                else: # node.val != p_val+1, restart
                    acc_path = 1
                    max_path = max(max_path, p_acc_path)

                # leaf node
                if not node.left and not node.right:
                    max_path = max(max_path, acc_path)
                    continue

                if node.left:
                    q.append((node.left, node.val, acc_path))

                if node.right:
                    q.append((node.right, node.val, acc_path))

            return max_path
          ```
      * Approach1-2: BFS Iterative, from the perspective of parent node Time:O(n), Space:O(n)
        * From the perspective of parent, can avoid unnecessary prev_val arg passing
        * python
          ```python
          def longestConsecutive(self, root: TreeNode) -> int:
            if not root:
                return 0

            # perspective of dummy node before root
            q = collections.deque([(root, 1)])
            max_path = 0
            while q:
              # current node, cur path
              node, path = q.popleft()
              max_path = max(max_path, path)

              for child in (node.left, node.right):
                  if not child:
                      continue

                  new_path = 1
                  if node.val+1 == child.val:
                      new_path = path+1

                  q.append((child, new_path))

          return max_path
          ```
      * Approach2: DFS Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def longestConsecutive(self, root: TreeNode) -> int:
            def dfs(node, path):
                nonlocal max_path
                max_path = max(max_path, path)
                for child in (node.left, node.right):
                  if not child:
                      continue

                  new_path = 1
                  if node.val + 1 == child.val:
                      new_path = path + 1
                  dfs(child, new_path)

            if not root:
                return 0

            max_path = 0
            dfs(root, 1)
            return max_path
          ```
    * 331: Verify Preorder Serialization of a Binary Tree	(M)
  * **Inorder**
    * 094: Binary Tree **Inorder** Traversal (M)
      * Approach1: Recursive, Time:O(n), Space:O(n):
        * Python
          ```python
          def inorderTraversal(self, root: TreeNode) -> List[int]:
            def _inorder(node):
                if not node:
                    return

                _inorder(node.left)
                output.append(node.val)
                _inorder(node.right)


            if not root:
                return []

            output = []
            _inorder(root)
            return output
          ```
      * Approach2: Iterative, Time:O(n), Space:O(n):
        * Python
          ```python
          def inorderTraversal(self, root: TreeNode) -> List[int]:
            if not root:
                return []

            output = []
            stack = []
            cur = root
            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left
                else:
                    cur = stack.pop()
                    output.append(cur.val)
                    cur = cur.right

            return output
          ```
    * 173: Binary Search Tree **Iterator** (M) *
      * Approach1: In-Order Iterative
        * Python Solution
          ```python
          class BSTIterator:
            def __init__(self, root: TreeNode):
                self.stack = list()
                self._push_all(root)

            def next(self) -> int:
                """
                @return the next smallest number
                """
                cur = self.stack.pop()
                self._push_all(cur.right)
                return cur.val

            def hasNext(self) -> bool:
                """
                @return whether we have a next smallest number
                """
                return len(self.stack) != 0

            def _push_all(self, cur: TreeNode):
                while cur:
                    self.stack.append(cur)
                    cur = cur.left
          ```
      * Approach1: Recursive
        * Python Solution
          ```python
          def inorderTraversal(self, root: TreeNode) -> List[int]:
            def _inorder(node):
                if not node:
                    return

                _inorder(node.left)
                output.append(node.val)
                _inorder(node.right)

            output = []
            _inorder(root)
            return output
          ```
      * Approach2: Iterative
        * Python Solution
          ```python
          def inorderTraversal(self, root: TreeNode) -> List[int]:
            if not root:
                return []

            output = []
            stack = []
            cur = root
            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left
                else:
                    cur = stack.pop()
                    output.append(cur.val)
                    cur = cur.right

            return output
          ```
    * 105: Construct Binary Tree from **Preorder** and **Inorder** Traversal (M)
      * Preorder -> find the root node (from left)
      * Inorder -> find the left subtree and right subtree of each root
      * Approach1: DFS Recursive (first version), Time:O(n^2) Space:O(n)
        * Python, Time:O(n^2) Space:O(n)
          * Time:
            * Find the partition costs O(n) for n nodes
          * Space:
            * Call stack and copy the inorder , preorder list cost O(n)
          ```python
          def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
            def get_subtree_partition(inorder, root_val):
                # partition is the idx of thr root node
                for idx, val in enumerate(inorder):
                    if val == root_val:
                        return idx

                return None

            def build_tree(preorder, inorder):
                if not preorder:
                    return None

                root_val = preorder[0]
                root = TreeNode(root_val)
                partition = get_subtree_partition(inorder, root_val)
                if partition is None:
                    return None

                left_inorder = inorder[:partition]
                left_preorder = preorder[1:len(left_inorder)+1]
                root.left = build_tree(left_preorder, left_inorder)

                right_inorder = inorder[partition+1:]
                right_preorder = preorder[len(left_inorder)+1:]
                root.right = build_tree(right_preorder, right_inorder)

                return root

            if not preorder or not inorder:
                return None

            return build_tree(preorder, inorder)
          ```
      * Approach2: DFS Recursive (optimization), Time:O(n), Space:O(n)
        * Use hash table to speed up the searching
        * Prevent unused list copy
        * Time:O(n), Space:O(n)
          * Python
            ```python
            def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
              def build_tree(left, right):
                  if left > right:
                      return None

                  nonlocal pre_idx
                  root_val = preorder[pre_idx]
                  partition_idx = inorder_d[root_val]
                  root= TreeNode(root_val)

                  pre_idx += 1 # <----- use this to control preorder

                  root.left = build_tree(left, partition_idx-1)
                  root.right = build_tree(partition_idx+1, right)

                  return root

              if not preorder:
                  return None

              inorder_d = dict()
              for idx, c in enumerate(inorder):
                  inorder_d[c] = idx

              pre_idx = 0

              return build_tree(0, len(inorder)-1)
            ```
      * Approach3: Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
            if not preorder or not inorder:
                return None

            inorder_d = dict()
            for idx, c in enumerate(inorder):
                inorder_d[c] = idx

            pre_idx = 0
            root = TreeNode(0)

            stack = [(root, 0, len(inorder)-1)]

            while stack:
                node, left, right = stack.pop()
                val = preorder[pre_idx]
                node.val = val
                partition = inorder_d[val]

                pre_idx += 1 # control the cur root idx of preorder

                if partition+1 <= right:
                    node.right = TreeNode(0)
                    stack.append((node.right, partition+1, right))

                # finish left subtree first (of pre_idx would be wrong)
                if left <= partition-1:
                    node.left = TreeNode(0)
                    stack.append((node.left, left, partition-1))

            return root
          ```
    * 106: Construct Binary Tree from **Inorder** and **Postorder** Traversal	(M)
      * Preorder -> find the root node (from right)
      * Inorder -> find the left subtree and right subtree of each root
      * Approach1: Recurisve, Time:O(n), Space:O(n)
        * Python
          ```python
          def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
            def build_tree(left, right):
                if left > right:
                    return None

                nonlocal post_idx
                root_val = postorder[post_idx]
                root = TreeNode(root_val)
                partition = inorder_d[root_val]

                post_idx -= 1
                # complete the right subtree first
                root.right = build_tree(partition+1, right)
                root.left = build_tree(left, partition-1)

                return root

            if not inorder or not postorder:
                return None

            post_idx = len(postorder) - 1

            inorder_d = dict()
            for idx, c in enumerate(inorder):
                inorder_d[c] = idx

            return build_tree(0, len(inorder)-1)
          ```
      * Approach2: Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:

            if not inorder or not postorder:
                return None

            post_idx = len(postorder) - 1

            inorder_d = dict()
            for idx, c in enumerate(inorder):
                inorder_d[c] = idx

            root = TreeNode(0)
            stack = [(root, 0, len(inorder)-1)]

            while stack:
                node, left, right = stack.pop()
                val = postorder[post_idx]
                node.val = val
                partition = inorder_d[val]

                post_idx -= 1

                if left <= partition-1:
                    node.left = TreeNode(0)
                    stack.append((node.left, left, partition-1))

                # complete the right subtree first
                if partition+1 <= right:
                    node.right = TreeNode(0)
                    stack.append((node.right, partition+1, right))

            return root
          ```
  * **Postorder**
    * 145: Binary Tree Postorder **Traversal** (H)
      * Approach1: Recursive, Time:O(n), Space:O(n)
        * Python Solution
            ```python
            def postorderTraversal(self, root: TreeNode) -> List[int]:
              def _postorder(node):
                  if not node:
                      return

                  _postorder(node.left)
                  _postorder(node.right)
                  output.append(node.val)

              output = []
              _postorder(root)
              return output
          ```
      * Approach2: Iterative, from end to start, Time:O(n), Space:O(n)
        * traverse order: root -> right -> left (use deque)
        * Python Solution
            ```python
              def postorderTraversal(self, root: TreeNode) -> List[int]:
                if not root:
                    return

                stack = [root]
                # reverse order
                output = collections.deque()

                while stack:
                    node = stack.pop()
                    output.appendleft(node.val)

                    if node.left:
                        stack.append(node.left)
                    if node.right:
                        stack.append(node.right)

                return output
            ```
      * Approach3: Iterative, from start to end, Time:O(n), Space:O(n)
        * traverse order: left -> right -> root
        * Python
          ```python
          def postorderTraversal(self, root: TreeNode) -> List[int]:
            if not root:
                return

            stack = []
            output = []
            prev_traverse = None
            cur = root

            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left
                else:
                    top = stack[-1]
                    # top.right != prev_traverse means top.right has not traversed yet
                    if top.right and top.right is not prev_traverse:
                        cur = top.right

                    else:
                        output.append(top.val)
                        prev_traverse = stack.pop()

            return output
          ```
      * Approach4: Morris Traversal, Time:O(n), Space:O(1)
    * 110: **Balanced** Binary Tree (E)
      * A binary tree in which the depth of the two subtrees of **every node** never differ by more than 1.
      * Even left subtree and right subtree are balanced trees, the tree may not be balanced tree.
      * Approach1-1: Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def isBalanced(self, root: TreeNode) -> bool:
            def dfs(node, depth):
                depth += 1
                left_d = right_d = depth

                # check left subtree
                if node.left:
                    is_balanced, left_d = dfs(node.left, depth)
                    if not is_balanced:
                        return False, -1 # -1 means don't care

                # check right subtree
                if node.right:
                    is_balanced, right_d = dfs(node.right, depth)
                    if not is_balanced:
                        return False, -1

                # check itself
                return abs(left_d-right_d) <= 1, max(left_d, right_d)

            if not root:
                return True

            is_balanced, _ = dfs(root, 0)

            return is_balanced
          ```
      * Approach1-2: Recursive, Time:O(n), Space:O(n)
        * Bottom up len, prevent unnecessary depth arg
          * Python
            ```python
            def isBalanced(self, root: TreeNode) -> bool:
              def dfs(node):
                  if not node:
                      return True, 0

                  is_balanced, left_d = dfs(node.left)
                  if not is_balanced:
                    return False, -1  # -1 means don't care

                  is_balanced, right_d = dfs(node.right)
                  if not is_balanced:
                    return False, -1

                  # check the node itself
                  return abs(left_d-right_d)<=1, 1+max(left_d, right_d)

              if not root:
                  return True

              return dfs(root)[0]
            ```
      * Approach2: Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def isBalanced(self, root: TreeNode) -> bool:
            if not root:
                return True

            post_order = collections.deque()
            stack = [root]
            # create post order list
            while stack:
                node = stack.pop()
                post_order.appendleft(node)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            memo = collections.defaultdict(int)
            is_balanced = True
            # post order traversal
            for node in post_order:
                # get depth of left subtree and right subtree
                left_d, right_d = memo[node.left], memo[node.right]
                if abs(left_d - right_d) > 1:
                    is_balanced = False
                    break
                # save tree depth to the memo
                memo[node] = 1+ max(left_d, right_d)
            return is_balanced
          ```
    * 124: Binary Tree **Maximum Path Sum** (H)
      * A path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
      * A "path" is like unicursal, **can't come back**
      * There are two cases for each node:
        * case1: do not connect with parent, the node can connect with two children.
        * case2: connect with parent, the node can connect with at most one child.
      * Approach1: Iterative
        * Python
          ```python
          def maxPathSum(self, root: TreeNode) -> int:
            if not root:
                return 0

            # create post order list
            stack = [root]
            post_order = collections.deque()
            while stack:
                node = stack.pop()
                post_order.appendleft(node)
                if node.left:
                    stack.append(node.left)

                if node.right:
                    stack.append(node.right)

            # traverse the post order and calculate the max sum path for each node
            max_sum = -float('inf')
            memo = collections.defaultdict(int)
            for node in post_order:
                s = node.val
                # get max left path and max right path
                max_left = max(memo[node.left], 0)
                max_right = max(memo[node.right], 0)

                # case1: do not connect with parent, the node can connect with two children.
                max_sum = max(max_sum, s+max_left+max_right)

                # case2: connect with parent, the node can connect with at most one child.
                memo[node] = s+max(max_left, max_right)

            return max_sum
          ```
      * Approach2: Recursive
        * Python
          ```python
          def maxPathSum(self, root: TreeNode) -> int:
            def dfs(node):
                if not node:
                    return 0

                s = node.val
                max_left = max(dfs(node.left), 0)
                max_right = max(dfs(node.right), 0)

                # case1: do not connect with parent, the node can connect with two children.
                nonlocal max_sum
                max_sum = max(max_sum, s+max_left+max_right)

                # case2: connect with parent, the node can connect with at most one child.
                return s+max(max_left, max_right)

            if not root:
                return 0

            max_sum = -float('inf')
            dfs(root)
            return max_sum
          ```
    * 250: Count **Univalue Subtrees** (M)
      * A Uni-value subtree means all nodes of the subtree have the same value.
      * Recursive:
        * Python
          ```python
          def countUnivalSubtrees(self, root: TreeNode) -> int:
            def dfs(node):
                nonlocal cnt
                # leaf node
                if not node.left and not node.right:
                    cnt += 1
                    return True

                is_univalue = True
                for child in (node.left, node.right):
                    if not child:
                        continue
                    if not dfs(child) or node.val != child.val:
                        is_univalue = False
                        # can not break here, since subtree may be uni-value

                if is_univalue:
                    cnt += 1
                return is_univalue

            if not root:
                return 0
            cnt = 0
            dfs(root)
            return cnt
          ```
      * Iterative:
        * Python
          ```python
          def countUnivalSubtrees(self, root: TreeNode) -> int:
            if not root:
                return 0

            stack = [root]
            post_order = collections.deque()
            while stack:
                node = stack.pop()
                post_order.appendleft(node)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            memo = dict()
            cnt = 0
            for node in post_order:
                # leaf node
                if not node.left and not node.right:
                    cnt += 1
                    memo[node] = True
                    continue

                is_univalue = True
                for child in (node.left, node.right):
                    if not child:
                        continue

                    if not memo[child] or node.val != child.val:
                        is_univalue = False
                        break

                if is_univalue:
                    cnt += 1

                memo[node] = is_univalue

            return cnt
        ```
    * 687. Longest **Univalue Path** (E)
      * Similar to 124
      * Recursive:
        * Python
          ```python
          def longestUnivaluePath(self, root: TreeNode) -> int:
            def dfs(node):
                nonlocal max_cnt
                # leaf node
                if not node.left and not node.right:
                    return 1

                left_cnt = right_cnt = 0
                if node.left:
                    cnt = dfs(node.left)
                    if node.val == node.left.val:
                        left_cnt = cnt

                if node.right:
                    cnt = dfs(node.right)
                    if node.val == node.right.val:
                        right_cnt = cnt

                # case1: do not connect to parent node
                max_cnt = max(max_cnt, 1+left_cnt+right_cnt)

                # case2: connect to parent node
                return 1+max(left_cnt,right_cnt)


            if not root:
                return 0

            max_cnt = 1

            dfs(root)

            return max_cnt - 1
          ```
      * Iterative:
        * Python
          ```python
          def longestUnivaluePath(self, root: TreeNode) -> int:

            if not root:
                return 0

            stack = [root]
            post_order = collections.deque()
            while stack:
                node = stack.pop()
                post_order.appendleft(node)

                if node.left:
                    stack.append(node.left)

                if node.right:
                    stack.append(node.right)

            max_cnt = 1
            memo = dict()
            for node in post_order:
                if not node.left and not node.right:
                    memo[node] = 1
                    continue

                left_cnt = right_cnt = 0
                if node.left:
                    cnt = memo[node.left]
                    if node.val == node.left.val:
                        left_cnt = cnt

                if node.right:
                    cnt = memo[node.right]
                    if node.val == node.right.val:
                        right_cnt = cnt

                # case1, do not connect to parent
                max_cnt = max(max_cnt, 1+left_cnt+right_cnt)

                # case2, connect to parent
                memo[node] = 1 + max(left_cnt, right_cnt)

            return max_cnt-1
          ```
    * 366: **Find Leaves** of Binary Tree (M)
      * Recursive:
        * Python
          ```python
          def findLeaves(self, root: TreeNode) -> List[List[int]]:
            def dfs(node):
              # case1: empty node -> level == -1
              # case2: non leaf node-> level == max_child level + 1
              if not node:
                  return -1

              level = 1 + max(dfs(node.left), dfs(node.right))

              if len(output) < level + 1:
                  output.append([])
              output[level].append(node.val)

              return level

            if not root:
              return []

            output = []
            dfs(root)
            return output
          ```
      * Iterative:
        * Python
          ```python
          def findLeaves(self, root: TreeNode) -> List[List[int]]:
            if not root:
                return []

            stack = [root]
            post_order = collections.deque()
            while stack:
                node = stack.pop()
                post_order.appendleft(node)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            output = []
            memo = collections.defaultdict(lambda: -1)
            for node in post_order:
                level = 1 + max(memo[node.left], memo[node.right])
                memo[node] = level

                if len(output) < level + 1:
                    output.append([])
                output[level].append(node.val)

                return output
          ```
    * 337: House Robber III (M)
      * See DP
    * 236: **Lowest Common Ancestor** of a **Binary Tree** (M)
      * Algo:
        * https://leetcode.com/articles/lowest-common-ancestor-of-a-binary-tree/
      * Three cases to find the LCA:
        * mid + left
        * mid + right
        * left + right
      * Approach1: postorder recursive, Time:O(n), Space:O(n)
        * algo:
          * Start traversing the tree from the root node.
          * If either of the left or the right branch returns True, this means one of the two nodes was found below.
          * If the current node itself is one of p or q, we would mark a variable mid as True and continue the search for the other node in the left and right branches.
          * If at any point in the traversal, any two of the three flags left, right or mid become True, this means we have found the lowest common ancestor for the nodes p and q.
        * Python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            def dfs(node):
                if not node:
                    return False

                mid = True if node is p or node is q else False
                left = dfs(node.left)
                right = dfs(node.right)

                """
                mid + left
                mid + right
                left + right
                """
                if mid + left + right >=2:
                    nonlocal lca
                    lca = node

                return mid or left or right

            if not root:
                return None

            lca = None
            dfs(root)
            return lca
          ```
      * Approach2: postorder iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            if not root:
                return None

            stack = [root]
            post_order = collections.deque()
            while stack:
                node = stack.pop()
                post_order.appendleft(node)

                if node.left:
                    stack.append(node.left)

                if node.right:
                    stack.append(node.right)

            memo = collections.defaultdict(bool)
            lca = None
            for node in post_order:
                mid = True if node is p or node is q else False
                left = memo[node.left]
                right = memo[node.right]

                """
                mid + left
                mid + right
                left + right
                """
                if mid + left + right >= 2:
                    lca = node
                    break

                memo[node] = mid or left or right

            return lca
          ```
    * 114: **Flatten** Binary Tree to Linked List	(M)
      * Right->Left->Root
      * Recursive, Time:O(n), Space:O(n)
        * https://leetcode.com/problems/flatten-binary-tree-to-linked-list/submissions/
        * Python
          ```python
          def flatten(self, root: TreeNode) -> None:
            def _flatten(node):
                if not node:
                    return

                _flatten(node.right)
                _flatten(node.left)

                nonlocal prev
                node.right = prev
                node.left = None
                prev = node    # prev pointer to flattern list

            if not root:
                return

            prev = None
            _flatten(root)
          ```
      * Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def flatten(self, root: TreeNode) -> None:
            if not root:
                return

            cur = root
            stack = []
            prev_traversed = None

            # right -> left -> root
            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.right
                else:
                    top = stack[-1]
                    if top.left and top.left is not prev_traversed:
                        cur = top.left
                    else:
                        node = stack.pop()
                        node.right = prev_traversed
                        node.left = None
                        prev_traversed = node
          ```
  * **BFS & DFS**
    * 101: **Symmetric** Tree (E)
      * Approach1: DFS Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def isSymmetric(self, root: TreeNode) -> bool:
            def is_symmetric(left, right):
                # None
                if not left and not right:
                    return True

                # 1 node
                if not left or not right:
                    return False

                if left.val != right.val:
                    return False

                return is_symmetric(left.left, right.right) and \
                       is_symmetric(left.right, right.left)

            if not root:
                return True

            return is_symmetric(root, root)
          ```
      * Approach2: BFS Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def isSymmetric(self, root: TreeNode) -> bool:
            if not root:
                return True

              q = collections.deque([(root.left, root.right)])
              is_symmetric = True
              while q:
                  left, right = q.popleft()

                  # 0 node
                  if not left and not right:
                      continue

                  # 1 node
                  if not left or not right:
                      is_symmetric = False
                      break

                  if left.val != right.val:
                      is_symmetric = False
                      break

                  q.append((left.left, right.right))
                  q.append((left.right, right.left))

              return is_symmetric
          ```
    * 257: Binary Tree **Paths** (E)
      * Each node has three cases:
        * no children: here is the end of the path
        * having left child: having left path
        * having right child: having left path
      * Approach1: DFS Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def binaryTreePaths(self, root: TreeNode) -> List[str]:

            def tree_path(node, path):
                path.append(str(node.val))
                # case1: no children, here is the end of the path
                if not node.left and not node.right:
                    output.append('->'.join(path))
                else:
                    # case2: having left path
                    if node.left:
                        tree_path(node.left, path)

                    # case3: having right path
                    if node.right:
                        tree_path(node.right, path)

                path.pop()

            if not root:
                return []

            output = []
            path = []
            tree_path(root, path)
            return output
          ```
      * Approach2: DFS Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def binaryTreePaths(self, root: TreeNode) -> List[str]:
            if not root:
                return []

            output = []
            stack = [(root, [])]
            while stack:
                node, path = stack.pop()
                path.append(str(node.val))

                # no children
                if not node.left and not node.right:
                    output.append('->'.join(path))

                # having right child
                if node.right:
                    stack.append((node.right, path))

                # having left child
                if node.left:
                    stack.append((node.left, path[:]))

            return output
          ```
    * 112: Path **Sum** (E)
      * Approach1: BFS, Time:O(n), Space:O(n)
        * Python
          ```python
          def hasPathSum(self, root: TreeNode, sum: int) -> bool:
            if not root:
                return False

            target = sum
            res = False
            q = collections.deque([(root, 0)])

            while q:
                node, acc = q.popleft()
                acc += node.val

                # no child
                if acc == target and not node.left and not node.right:
                    res = True
                    break

                if node.left:
                    q.append((node.left, acc))

                if node.right:
                    q.append((node.right, acc))

            return res
          ```
      * Approach2: DFS Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def hasPathSum(self, root: TreeNode, sum: int) -> bool:
            def has_path_sum(node, acc):
              if not node:
                  return False

              acc += node.val

              if acc == sum and not node.left and not node.right:
                  return True

              return has_path_sum(node.left, acc) or has_path_sum(node.right, acc)

              if not root:
                  return False

              return has_path_sum(root, 0)
          ```
      * Approach3: DFS Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def hasPathSum(self, root: TreeNode, sum: int) -> bool:
            if not root:
                return False

            target = sum
            res = False
            s = [(root, 0)]

            while s:
                node, acc = s.pop()
                acc += node.val

                # no child
                if acc == target and not node.left and not node.right:
                    res = True
                    break

                if node.left:
                    s.append((node.left, acc))

                if node.right:
                    s.append((node.right, acc))

            return res
          ```
    * 113: Path **Sum** II (M)
      * Approach1: BFS, Time:O(n), Space:O(n^2)
        * Space:
          * Since we keep a list in each queue element
        * Python
          ```python
          def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
            if not root:
                return []

            target = sum
            output = []
            # node, path, accumulated value
            q = collections.deque([(root, [], 0)])

            while q:
                node, path, acc = q.popleft()
                acc += node.val
                path.append(node.val)

                if acc == target and not node.left and not node.right:
                    output.append(path[:])

                if node.left:
                    q.append((node.left, path, acc))

                if node.right:
                    q.append((node.right, path[:], acc))

            return output
          ```
      * Approach2: DFS, Time:O(n), Space:O(n)
        * Python
          ```python
          def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
              def dfs(node, path, acc):
                  path.append(node.val)
                  acc += node.val

                  if acc == sum and not node.left and not node.right:
                      output.append(path[:])

                  else:
                      if node.left:
                          dfs(node.left, path, acc)

                      if node.right:
                          dfs(node.right, path, acc)

                  path.pop()

              if not root:
                  return []

              output = []
              dfs(root, [], 0)
              return output
          ```
      * Approach3: DFS Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
            if not root:
                return []

            target = sum
            output = []
            # node, path, accumulated value
            s = [(root, [], 0)]

            while s:
                node, path, acc = s.pop()
                acc += node.val
                path.append(node.val)

                if acc == target and not node.left and not node.right:
                    output.append(path[:])

                if node.left:
                    s.append((node.left, path[:], acc))

                if node.right:
                    s.append((node.right, path, acc))

            return output
          ```
    * 129: Sum Root to Leaf Numbers (M)
      * Approach1: BFS Iterative, Time:O(n), Space:O(n)
        ```python
        def sumNumbers(self, root: TreeNode) -> int:
          if not root:
              return 0

          q = collections.deque([(root, 0)])
          sum_leaf = 0
          while q:
              node, s = q.popleft()
              s += node.val

              if not node.left and not node.right:
                  sum_leaf += s

              if node.left:
                  q.append((node.left, s*10))

              if node.right:
                  q.append((node.right, s*10))

          return sum_leaf
        ```
      * Approach2: DFS Recursive, Time:O(n), Space:O(n)
        ```python
        def sumNumbers(self, root: TreeNode) -> int:
          def dfs(node, s):
              nonlocal sum_leaf
              s += node.val

              if not node.left and not node.right:
                  sum_leaf += s
                  return

              if node.left:
                  dfs(node.left, s*10)

              if node.right:
                  dfs(node.right, s*10)

            if not root:
                return 0

            sum_leaf = 0
            dfs(root, 0)
            return sum_leaf

        ```
    * 102: Binary Tree Level Order Traversal (E)
      * Approach1: DFS Recursive, Time:O(n), Space:O(n)
        * Python Solution
          ```python
          def _level_order(node, level):
            if not node:
                return

            if len(visits) < (level+1):
                for _ in range((level+1)-len(visits)):
                    visits.append([])

            visits[level].append(node.val)
            _level_order(node.left, level+1)
            _level_order(node.right, level+1)

            if not root:
                return []

            visits = []
            _level_order(root, level=0)
            return visits
          ```
      * Approach2: BFS Iterative, Time:O(n), Space:O(n)
        * Python Solution
          ```python
          def levelOrder(self, root: TreeNode) -> List[List[int]]:
            if not root:
                return []

            visits = []
            q = collections.deque([root])

            while q:
                q_len = len(q)
                cur_visits = []
                visits.append(cur_visits)

                for _ in range(q_len):
                    node = q.popleft()
                    cur_visits.append(node.val)

                    if node.left:
                        q.append(node.left)

                    if node.right:
                        q.append(node.right)

              return visits
          ```
    * 107: Binary Tree Level Order Traversal II (E)
      * Bottom up
      * Approach1: DFS Recursive, Time:O(n), Space:O(n)
        ```python
          def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
          if not root:
              return []

            def _level_order_bottom(node, level):

                if not node:
                    return

                if len(visits) < level + 1:
                    visits.appendleft([])

                # len(visits)-1 Is the right mode index
                visits[len(visits)-1-level].append(node.val)

                _level_order_bottom(node.left, level+1)
                _level_order_bottom(node.right, level+1)

            visits = collections.deque()
            _level_order_bottom(root, level=0)
            return visits
        ```
      * Approach2: BFS Iterative, Time:O(n), Space:O(n)
        * Python Solution
          ```python
          def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
            if not root:
                return []

            visits = collections.deque()
            q = collections.deque([root])

            while q:
                q_len = len(q)
                cur_visits = []
                visits.appendleft(cur_visits)
                for _ in range(q_len):
                    node = q.popleft()
                    cur_visits.append(node.val)

                    if node.left:
                        q.append(node.left)

                    if node.right:
                        q.append(node.right)

            return visits
          ```
    * 637: Average of Levels in Binary Tree (E)
      * Approach1 BFS: Time:O(n), Space:O(n)
        * Python Solution
          ```python
          def averageOfLevels(self, root: TreeNode) -> List[float]:
            if not root:
                return []

            output = []
            q = collections.deque([root])
            while q:
                level_cnt = len(q)
                acc = 0

                for _ in range(level_cnt):
                    node = q.popleft()
                    acc += node.val
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)

                output.append(acc/level_cnt)
            return output
          ```
      * Approach2 DFS
    * 103: Binary Tree **Zigzag** Level Order Traversal (M)
      * Approach1: BFS, Time:O(n), Space:O(n)
        * Python Solution:
          ```python
          def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
            if not root:
                return []

            visits = []
            q = collections.deque([root])
            append_left = False
            while q:
                q_len  = len(q)
                cur_visits = collections.deque()
                visits.append(cur_visits)

                for _ in range(q_len):
                    node = q.popleft()
                    if append_left:
                        cur_visits.appendleft(node.val)
                    else:
                        cur_visits.append(node.val)

                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)

                append_left = not append_left

            return visits
          ```
      * Approach2: DFS, Time:O(n), Space:O(n)
        * Python Solution
          ```python
          def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
            def _zigzag_level_order(node, level):
                if not root:
                    return

                if len(visits) < level + 1:
                    visits.append(collections.deque())

                if level % 2:
                    visits[level].appendleft(node.val)
                else:
                    visits[level].append(node.val)

                if node.left:
                    _zigzag_level_order(node.left, level+1)
                if node.right:
                    _zigzag_level_order(node.right, level+1)

            if not root:
                return []

            visits = []
            _zigzag_level_order(root, 0)
            return visits
          ```
    * 199: Binary Tree **Right Side** View (M)
      * Approach1 BFS
        * Python Solution
          ```python
          def rightSideView(self, root: TreeNode) -> List[int]:
            if not root:
                return []

            visits = []
            q = collections.deque([root])

            while q:
                q_len = len(q)
                cur = None
                for _ in range(q_len):
                    cur = q.popleft()
                    if cur.left:
                        q.append(cur.left)
                    if cur.right:
                        q.append(cur.right)

                visits.append(cur.val)
            return visits
          ```
      * Approach2 DFS Recursive:
        * Python Solution
          ```python
          def rightSideView(self, root: TreeNode) -> List[int]:
            def right_side_view(node, level):
                if not node:
                    return

                if len(visits) < level + 1:
                    visits.append(node.val)

                # right first
                if node.right:
                    right_side_view(node.right, level+1)

                if node.left:
                    right_side_view(node.left, level+1)

            if not root:
                return []

            visits = []
            right_side_view(root, 0)
            return visits
          ```
    * 314: Binary Tree **Vertical Order** Traversal	(M)
      * Node in the same vertical order should be in the same group (from top to down)
      * List group from left to right
      * Approach1: BFS, Time:(n), Space:(n)
        * BFS for top-down traversal of each group
        * Sort the key of the dic to output from left to right
        * Time: (nlog(n))
          * BFS takes O(n)
          * Sort group costs O(nlogn)
        * Space:
          * Queue: O(n)
          * Dictionary: O(n)
        * Python:
          ```python
          def verticalOrder(self, root: TreeNode) -> List[List[int]]:
            if not root:
                return []

            d = collections.defaultdict(list)
            # node and group index
            q = collections.deque([(root, 0)])

            while q:
                node, idx = q.popleft()
                d[idx].append(node.val)
                if node.left:
                    q.append((node.left, idx-1))

                if node.right:
                    q.append((node.right, idx+1))

            return [d[key] for key in sorted(d.keys())]
          ```
    * 297: **Serialize** and **Deserialize** Binary Tree (H)
      * Approach1: BFS + pickle, Time:O(n), Space:O(n)
        * Python
          ```python
          class Codec:
              def serialize(self, root):
                  """Encodes a tree to a single string.

                  :type root: TreeNode
                  :rtype: str
                  """
                  if not root:
                      return []

                  q = collections.deque([root])
                  bfs = []
                  while q:
                      node = q.popleft()
                      if not node:
                          # insert None to list can help to deserialize the tree
                          bfs.append(None)
                          continue

                      bfs.append(node.val)
                      q.append(node.left)
                      q.append(node.right)

                  return pickle.dumps(bfs)

              def deserialize(self, data):
                  """Decodes your encoded data to tree.

                  :type data: str
                  :rtype: TreeNode
                  """
                  if not data:
                      return None

                  bfs = pickle.loads(data)

                  if not bfs:
                      return None

                  root = TreeNode(bfs[0])
                  i = 1
                  q = collections.deque([root])

                  while q:
                      node = q.popleft()
                      # left
                      if bfs[i] != None:
                          node.left = TreeNode(bfs[i])
                          q.append(node.left)
                      i += 1  # move forward even bfs[i] is None

                      # right
                      if bfs[i] != None:
                          node.right = TreeNode(bfs[i])
                          q.append(node.right)
                      i += 1

                  return root
           ```
      * Approach2: DFS + pickle
    * 116 & 117: **Populating Next Right Pointers** in Each Node (M)
      * A perfect binary tree
        * where all leaves are on the same level, and every parent has two children.
      * Approach1: Use queue, Time:O(n), Space:O(n)
        * Python
          ```python
          def connect(self, root: 'Node') -> 'Node':

          if not root:
              return root

          q = collections.deque([root])
          dummy = Node(0)
          while q:
              q_len = len(q)

              prev = dummy
              for _ in range(q_len):
                  cur = q.popleft()
                  if cur.left:
                      q.append(cur.left)
                  if cur.right:
                      q.append(cur.right)

                  prev.next = cur
                  prev = cur

          return root
          ```
      * Approach2: Use next pointer, perfect binary tree, Time:O(n), Space:O(1)
        * Python
          ```python
          def connect(self, root: 'Node') -> 'Node':
            if not root:
                return None

            head = root
            # head.left means the start of next level
            # traverse from 2nd level
            # this solution is for perfect binary tree only
            while head and head.left:
                cur_root = head
                while cur_root:
                    cur_root.left.next = cur_root.right
                    cur_root.right.next = cur_root.next.left if cur_root.next else None

                    cur_root = cur_root.next

                head = head.left

            return root
          ```
      * Approach3: Use next pointer, general solution, Time:O(n), Space:O(1)
        * Python
          ```python
          def connect(self, root: 'Node') -> 'Node':
            if not root:
                return None

            prev = dummy = Node(0)
            head = root
            # traverse from 2nd level
            while head:
                cur_root = head
                while cur_root:
                    if cur_root.left:
                        prev.next = cur_root.left
                        prev = prev.next

                    if cur_root.right:
                        prev.next = cur_root.right
                        prev = prev.next

                    cur_root = cur_root.next

                head = dummy.next # dummy.next ponters to the head of next level
                prev = dummy      # init the dummy and prev again
                dummy.next = None

            return root
          ```
  * **Binary Search Tree** (BST)
    * Definition of BST:
      * The **left subtree** of a node contains only nodes with keys **less than** the node's key.
      * The **right subtree** of a node contains only nodes with keys **greater than** the node's key.
      * Both the left and right subtrees must also be binary search trees.
    * 098: **Validate** Binary Search Tree (M)
      * Approach1: Postorder, Recursive, Time:O(h), Space:O(h)
        * For each node
          * The maximum val of left subtree should be less than the node val
          * The minimum val of right subtree should be greater than the node val
          * that is : l_max < node.val < r_min
        * Python:
          ```python
          def isValidBST(self, root: TreeNode) -> bool:
            def dfs(node):
                if not node:
                    # is_valid, min_val, max_val
                    return True, float('inf'), float('-inf'),

                l_is_valid, l_min, l_max = dfs(node.left)
                if not l_is_valid:
                    return False, None, None # None means don't care

                r_is_valid, r_min, r_max = dfs(node.right)
                if not r_is_valid:
                    return False, None, None # None means don't care

                # is_valid: if the tree is a binary search tree
                # minimum value in the tree
                # maximum value in the tree
                return l_max < node.val < r_min, \
                        min(node.val, l_min, r_min), max(node.val, l_max, r_max)

            return dfs(root)[0]
          ```
      * Approach2: Preorder Recursive, Time:O(h), Space:O(h)
        * Python
          ```python
          def isValidBST(self, root: TreeNode) -> bool:
            def dfs(node, lower, upper):
                if not node:
                    return True

                if not lower < node.val < upper:
                    return False

                return dfs(node.left, lower, node.val) and dfs(node.right, node.val, upper)

            return dfs(root, float('-inf'), float('inf'))
          ```
      * Approach3: Preorder, Iterative, Time:O(h), Space:O(h)
        * Python
          ```python
          def isValidBST(self, root: TreeNode) -> bool:
            if not root:
                return True
            is_valid = True
            stack = [(root, float('-inf'), float('inf'))]
            while stack:
                node, lower, upper = stack.pop()
                if not lower < node.val < upper:
                    is_valid = False
                    break
                if node.left:
                    stack.append((node.left, lower, node.val))

                if node.right:
                    stack.append((node.right, node.val, upper))

            return is_valid
          ```
      * Approach4: InOrder Iterative, Time:O(h), Space:O(h)
        * Python
          ```python
          def isValidBST(self, root: TreeNode) -> bool:
            if not root:
                return True

            cur = root
            stack = []
            in_order = float('-inf')
            is_valid = True
            while stack or cur:
                if cur:
                    stack.append(cur)
                    cur = cur.left
                else:
                    cur = stack.pop()
                    if cur.val <= in_order:
                        is_valid = False
                        break
                    in_order = cur.val
                    cur = cur.right

            return is_valid
          ```
    * 701: **Insert** into a BST (M)
      * Recursive: Time:O(h), Space:O(h)
        * Python
          ```python
          def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
            def dfs(node):
                if val > node.val:
                    if node.right:
                        dfs(node.right)
                    else:
                        node.right = TreeNode(val)

                elif val < node.val:
                    if node.left:
                        dfs(node.left)
                    else:
                        node.left = TreeNode(val)

            if not root:
                return TreeNode(val)

            dfs(root)

            return root
          ```
      * Iterative: Time:O(h), Space:O(h)
        * Python
          ```python
          def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
            if not root:
                return TreeNode(val)

            cur = root
            prev = None
            while cur:
                prev = cur
                if val > cur.val:
                    if cur.right:
                        cur = cur.right
                    else:
                        cur.right = TreeNode(val)
                        break

                elif val < cur.val:
                    if cur.left:
                        cur = cur.left
                    else:
                        cur.left = TreeNode(val)
                        break
                else:
                    break

            return root
          ```
    * 450: **Delete** Node in a BST (M)
      * Definition
        * Predecessor:
          * maxium value in the left subtree
        * Successor:
          * minimum value in the right subtree
      * 2 Cases for parent node:
        * It is the root.
        * It is not the root.
      * 3 Cases for delete node
        * It is a leaf node
          * Return None to parent node
        * It has only one subtree
          * Return the one subtree to parent
        * It has two subtrees:
      * Recursive1: Time:O(h), Space:O(h)
        * Ref:
          * https://leetcode.com/articles/delete-node-in-a-bst/
        * Python
          ```python
          def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
            def get_predecessor(node):
                "maximum value in the left subtree"
                cur = node.left
                while cur.right:
                    cur = cur.right

                return cur

            def get_successor(node):
                "minimum value in the right subtree"
                cur = node.right
                while cur.left:
                    cur = cur.left

                return cur

            def dfs(node, val):

                if not node:  # not found
                    return None

                if val < node.val:
                    node.left = dfs(node.left, val)

                elif val > node.val:
                    node.right = dfs(node.right, val)

                else: # val == node.val:
                    if not node.left and not node.right:
                        return None  # leaf node

                    if not node.left:
                        return node.right

                    if not node.right:
                        return node.left

                    if node.right:
                        # minimum value in the right subtree
                        successor = get_successor(node)
                        node.val = successor.val
                        node.right = dfs(node.right, successor.val)

                    else:
                        # maximum value in the left subtree
                        predecessor = get_predecessor(node)
                        node.val = predecessor.val
                        node.left = dfs(node.left, predecessor.val)

                return node

            return dfs(root, key)
          ```
      * Recursive2: Time:O(h), Space:O(h)
        * Find the node:
          * case1: leaf node, return None to parent node
          * case2: no left subtree, return right subtree
          * case3: no right subtree, return left subtree
          * case4: node.right replaces the node, and append node.left to the left of the successor
        * Python
          ```python
          def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
            def get_successor(node):
                cur = node.right
                while cur.left:
                    cur = cur.left
                return cur

            def dfs(node):
                if not node:
                    return None # not found

                if key < node.val:
                    node.left = dfs(node.left)
                    return node

                elif key > node.val:
                    node.right = dfs(node.right)
                    return node

                else: # key == node.val

                    # leaf node
                    if not node.left and not node.right:
                        return None

                    if not node.left:
                        return node.right

                    if not node.right:
                        return node.left

                    right_min = get_successor(node)
                    right_min.left = node.left

                    # node.right replaces the node
                    return node.right

            if not root:
                return None

            return dfs(root)
          ```
      * Iterative2: Time:O(h), Space:O(h)
        * Find the node:
          * case1: leaf node, return None to parent node
          * case2: no left subtree, return right subtree
          * case3: no right subtree, return left subtree
          * case4: node.right replaces the node, and append node.left to the left of the successor
        * Python
        ```python
        def deleteNode(self, root: TreeNode, key: int) -> TreeNode:

          def get_successor(node):
              cur = node.right
              while cur.left:
                  cur = cur.left
              return cur

          def delete_node(node):

              # leaf node
              if not node.left and not node.right:
                  return None

              if not node.left:
                  return node.right

              if not node.right:
                  return node.left

              right_min = get_successor(node)
              right_min.left = node.left

              # node.right replace the node
              return node.right


          if not root:
              return None

          prev = None
          cur = root
          while cur:
              if key < cur.val:
                  prev = cur
                  cur = cur.left
              elif key > cur.val:
                  prev = cur
                  cur = cur.right
              else:  # cur.val == key
                  break

          if not cur:
              return root # not found

          if not prev:    # delete root node
              return delete_node(cur)

          if key < prev.val:
              prev.left = delete_node(cur)
          else:
              prev.right = delete_node(cur)

          return root
        ```
    * 235: **Lowest Common Ancestor** of a Binary Search Tree (E)
      * Algo:
        * https://leetcode.com/articles/lowest-common-ancestor-of-a-binary-search-tree/
        * 1: Start traversing the tree from the root node.
        * 2: If both the nodes p and q are in the right subtree, then continue the search with right subtree starting step 1.
        * 3: If both the nodes p and q are in the left subtree, then continue the search with left subtree starting step 1.
        * 4: If both step 2 and step 3 are not true, this means we have found the node which is common to node p's and q's subtrees. and hence we return this common node as the LCA.
          * cur + left
          * cur + right
          * left + right
      * FAQ
        * How to error handling if p and q are not in the tree ?? Use 236 (postorder)

      * Preorder Recursive, Time:O(n), Space:O(n)
        * python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            def dfs(node):
                val = node.val
                # find right subtree
                if node.val < p_val and node.val < q_val:
                    if node.right:
                        return dfs(node.right)
                    else:
                        return None
                elif node.val > p_val and node.val > q_val:
                    if node.left:
                        return dfs(node.left)
                    else:
                        return None
                else:
                    return node

            if not root:
                return None

            p_val = p.val
            q_val = q.val

            return dfs(root)
          ```
      * Preorder Iterative, Time:O(n), Space:O(1)
        * Python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            if not root:
                return None

            p_val = p.val
            q_val = q.val

            cur = root
            while cur:
                if cur.val < p_val and cur.val < q_val:
                    cur = cur.right

                elif cur.val > p_val and cur.val > q_val:
                    cur = cur.left

                else:
                    break

            return cur
          ```
    * 108: Convert **Sorted Array** to Binary Search Tree	binary search (E)
      * A **height-balanced binary** tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
      * Recursive: Time:O(n), Space:O(log(n))
        * Python
          ```python
          def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
            def dfs(node, left, right):
                mid = (left + right) // 2
                node.val = nums[mid]

                if left < mid:
                    node.left = TreeNode(0)
                    dfs(node.left, left, mid-1)

                if right > mid:
                    node.right = TreeNode(0)
                    dfs(node.right, mid+1, right)

            if not nums:
                return None

            root = TreeNode(0)
            dfs(root, 0, len(nums)-1)
            return root
          ```
      * Iterative: Time:O(n), Space:(log(n))
        * Python
          ```python
          def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
            if not nums:
                return None

            root = TreeNode(0)
            stack = [(root, 0, len(nums)-1)]

            while stack:
                node, left, right = stack.pop()
                mid = (left + right) // 2
                node.val = nums[mid]

                if left < mid:
                    node.left = TreeNode(0)
                    stack.append((node.left, left, mid-1))

                if right > mid:
                    node.right = TreeNode(0)
                    stack.append((node.right, mid+1, right))

            return root

          ```
    * 109: Convert **Sorted List** to Binary Search Tree binary search (M)
      * Approach1: Two Pointer to find middle, Time:O(nlog(n)), space:O(log(n))
        * Time: O(nlogn)
          * n/2 + 2*n/4 + 4*n/8 + ... logn * 1
        * This is not a good approach, since the input will be changed.
      * Approach2: Transfer to array, Time:O(log(n)), Space:O(n)
        * Pytyon
          ```python
          def sortedListToBST(self, head: ListNode) -> TreeNode:
            if not head:
                return None

            nums = []
            while head:
                nums.append(head.val)
                head = head.next

            left = 0
            right = len(nums)-1
            root = TreeNode(0)
            stack = [(root, left, right)]

            while stack:
                node, left, right = stack.pop()
                mid = (left + right)//2
                node.val = nums[mid]
                if left < mid:
                    node.left = TreeNode(0)
                    stack.append((node.left, left, mid-1))

                if right > mid:
                    node.right = TreeNode(0)
                    stack.append((node.right, mid+1, right))

            return root
          ```
      * Approach3: Inorder Simulation, Time:O(n), Space:O(log(n)
    * 230: **Kth Smallest** Element in a BST (M)
      * Approach1: Inorder Traversal, Time:O(h+k), Space:O(h)
        * Python
          ```python
          def kthSmallest(self, root: TreeNode, k: int) -> int:
            cur = root
            stack = []
            cnt = 0
            target = None
            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left
                else:
                    cur = stack.pop()
                    cnt += 1
                    if cnt == k:
                        target = cur.val
                        break

                    cur = cur.right

            return target
          ```
      * Follow up:
        * What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kth Smallest routine?
        * Doubly Linked List + Dict + Tree ??
    * 285: **Inorder Successor** in BST (M)
      * successor:
        * if the node with the smallest key greater than p.val
        * Successor is the smallest node in the inorder traversal after the current one.
      * Approach1: Inorder Traversal: Time:O(h), Space:O(h)
        * Python
          ```python
          def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
            if not root or not p:
                return None

            cur = root
            stack = []
            is_found = False
            successor = None
            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left

                else:
                    cur = stack.pop()

                    if is_found:
                        successor = cur
                        break

                    if cur is p:
                        is_found = True

                    cur = cur.right

            return successor
          ```
    * 510: **Inorder Successor** in BST II (M)
      * Two cases:
        * Have right subtree:
          * return minimum value of right subtree
        * Do not have right subtree
          * find the successor from ancestors
      * Approach1: Iterative: Time:O(h), Space:O(1)
        * Python
          ```python
          def inorderSuccessor(self, node: 'Node') -> 'Node':
            if not node:
                return None

            successor = None
            key = node.val

            # case1, have the right subtree, return the minimum value from right subtree
            if node.right:
                cur = node.right
                while cur.left:
                    cur = cur.left
                successor = cur

            # case2, do not have the right subtree, try to find from ancestors
            else:
                cur = node.parent
                while cur and cur.val < key:
                    cur = cur.parent
                if cur and cur.val > key:
                    successor = cur

            return successor
          ```
    * 270: **Closest** Binary Search Tree Value (E)
      * Inorder Recursive:
        * Python
          ```python
          def closestValue(self, root: TreeNode, target: float) -> int:
            def dfs(node):
                if node.left:
                    dfs(node.left)

                nonlocal closet
                if closet is None or abs(node.val-target) < abs(closet-target):
                    closet = node.val

                if node.right:
                    dfs(node.right)

            if not root:
                return None

            closet = NONE
            dfs(root)

            return closet
          ```
      * Inorder Iterative:
        * Python
          ```python
          def closestValue(self, root: TreeNode, target: float) -> int:
            if not root:
                return None

            cur = root
            stack = []
            closet = None
            found = False

            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left

                else:
                    cur = stack.pop()
                    # equal is for init cases
                    if closet is None or abs(cur.val-target) <= abs(closet-target):
                        closet = cur.val
                    else:
                        break
                    cur = cur.right

            return closet
          ```
    * 272: **Closest** Binary Search Tree Value II (H)
      * Approach1: Max Heap, Time:O(nlogk), Space:O(n)
        * Python
          ```python
          # for max heap
          class Element(object):
              def __init__(self, key, diff):
                  self.key = key
                  self.diff = diff

              def __lt__(self, other):
                  return self.diff > other.diff

              def __repr__(self):
                  return str(self.key)

          def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
              if not root or k < 1:
                  return []

              heap = []
              stack = []
              cur = root

              while cur or stack:
                  if cur:
                      stack.append(cur)
                      cur = cur.left
                  else:
                      cur = stack.pop()
                      diff = abs(target-cur.val)
                      if len(heap) < k:
                          heapq.heappush(heap, Element(cur.val, diff))
                      elif diff < heap[0].diff:
                          heapq.heapreplace(heap, Element(cur.val, diff))
                      else:
                          break
                      cur = cur.right

              return [e.key for e in heap]
          ```
      * Approach2: Deque, Time:O(n), Space:O(n)
        * There are 3 cases for diff sequences.
          * 1. 0 1 2 3 4 5
          * 2. 5 4 3 2 1 0
          * 3. 3 2 1 0 1 2 3
        * Python
          ```python
          def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
            if not root or k < 1:
                return []

            k_closets = collections.deque()
            stack = []
            cur = root
            while cur or stack:
                if cur:
                    stack.append(cur)
                    cur = cur.left
                else:
                    cur = stack.pop()
                    k_closets.append(cur.val)
                    cur = cur.right

            while len(k_closets) > k:
                # pop the bigger one
                if abs(k_closets[0]-target) >= abs(k_closets[-1]-target):
                    k_closets.popleft()
                else:
                    k_closets.pop()

            return k_closets
          ```
    * 096: **Unique** Binary Search **Trees**	(M)
      * Approach1: backtracking + memo, Time:O(n^2), Space:O(n^2)
        * Python
          ```python
          def numTrees(self, n: int) -> int:
            def back_track(left, right):
                """
                get cnt according to left and right subtree
                """
                if (left, right) not in memo:
                    # determine the left subtree and right subtree
                    r = l = 0
                    if left <= 1:
                        l = 1
                    else:
                        for i in range(left):
                            l += back_track(i, left-1-i)

                    if right <= 1:
                        r = 1
                    else:
                        for i in range(right):
                            r += back_track(i, right-1-i)

                    memo[(right, left)] = memo[(left, right)] = l*r

                return memo[(left, right)]

            if n < 1:
                return 0

            if n == 1:
                return 1

            cnt = 0
            memo = dict()
            memo[(0, 0)] = memo[(0, 1)] = memo[(1, 0)] = memo[(1, 1)] = 1

            # determine the root
            for i in range(n):
                cnt += back_track(i, n-1-i)
            return cnt
          ```
      * Approach2: DP, Time:O(n^2), Space:O(n)
        * g(n): the number of unique BST for a sequence of length n
          * g(0) = g(1) = 1
        * f(i, n): the number of unique BST, where the number i is served as the root of BST
        * Python
          ```python
          def numTrees(self, n: int) -> int:
            memo = [0] * (n + 1)
            memo[0] = memo[1] = 1

            # i from 2 to n
            for i in range(2, n+1):
                # total i nodes, 1 is for root
                # left + right = i-1
                left_right_cnt = i-1
                for left_cnt in range(0 , i):
                    memo[i] += memo[left_cnt] * memo[left_right_cnt-left_cnt]

            return memo[n]
          ```
      * Approach3: Mathematical Deduction, Time:O(n), Space:O(1)
        * [Catalan number](https://en.wikipedia.org/wiki/Catalan_number)
        * Python
          ```python
          def numTrees(self, n: int) -> int:
            catalan = 1

            for i in range(0, n):
                catalan = catalan * 2*(2*i+1)/(i+2)

            return int(catalan)
          ```
    * 095: **Unique** Binary Search **Trees** II (M)
    * 255: Verify Preorder Sequence in Binary Search Tree (M)
      * See Stack and Queue
    * 333: Largest BST Subtree (M)
    * 099: **Recover** Binary Search Tree (H)
  * **Other**:
    * 156: Binary Tree Upside Down (M)
### Trie (Prefix Tree)
  * 208: Implement Trie (M)
    * Approach1: Iterative
      * Insert: O(k)
      * Search: O(k)
      * Python Solution:
        ```python
        class Node(object):
          def __init__(self):
              self.children = collections.defaultdict(Node)
              self.end_of_word = False

        class Trie:
          def __init__(self):
              """
              Initialize your data structure here.
              """
              self.root = Node()

          def insert(self, word: str) -> None:
              """
              Inserts a word into the trie.
              """
              cur = self.root
              for c in word:
                  child = cur.children[c]
                  cur = child

              cur.end_of_word = True

          def search(self, word: str) -> bool:
              """
              Returns if the word is in the trie.
              """
              cur = self.root
              for c in word:
                  if c not in cur.children:
                      return False
                  cur = cur.children[c]

              return cur.end_of_word

          def startsWith(self, prefix: str) -> bool:
              """
              Returns if there is any word in the trie that starts with the given prefix.
              """
              cur = self.root
              for c in prefix:
                  if c not in cur.children:
                      return False
                  cur = cur.children[c]

              return True
        ```
    * Approach2: Recurisve:
      ```python
      import collections

      class Node(object):
          def __init__(self):
              self.children = collections.defaultdict(Node)
              self.end_of_word = False

      class Trie:

          def __init__(self):
              """
              Initialize your data structure here.
              """
              self.root = Node()

          def insert(self, word: str) -> None:
              """
              Inserts a word into the trie.
              """
              def insert_with_idx(cur, idx):
                  if idx == w_len:
                      cur.end_of_word = True
                      return

                  insert_with_idx(cur.children[word[idx]], idx+1)

              w_len = len(word)
              insert_with_idx(self.root, 0)


          def search(self, word: str) -> bool:
              """
              Returns if the word is in the trie.
              """
              def search_with_idx(cur, idx):
                  if idx == w_len:
                      return cur.end_of_word

                  c = word[idx]
                  if c not in cur.children:
                      return False
                  return search_with_idx(cur.children[c], idx+1)

              w_len = len(word)
              return search_with_idx(self.root, 0)


          def startsWith(self, prefix: str) -> bool:
              """
              Returns if there is any word in the trie that starts with the given prefix.
              """
              def search_with_idx(cur, idx):
                  if idx == w_len:
                      return True

                  c = prefix[idx]
                  if c not in cur.children:
                      return False
                  return search_with_idx(cur.children[c], idx+1)

              w_len = len(prefix)
              return search_with_idx(self.root, 0)
      ```
  * 211: Add and Search Word - Data structure design (M)
    * search word (**support wildcard**)
    * Approach1: Iterative:
      * Python Solution
        ```python
        class Node(object):
          def __init__(self):
              self.children = collections.defaultdict(Node)
              self.end_of_word = False

        class WordDictionary:

            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.root = Node()


            def addWord(self, word: str) -> None:
                """
                Adds a word into the data structure.
                """
                cur = self.root

                for c in word:
                    cur = cur.children[c]

                cur.end_of_word = True

            def search(self, word: str) -> bool:
                """
                Returns if the word is in the data structure.
                A word could contain the dot character '.' to represent any one letter.
                """
                stack = list()
                stack.append((self.root, 0))
                w_len = len(word)

                found = False
                while stack:
                    cur, idx = stack.pop()
                    # final node
                    if idx == w_len:
                        if cur.end_of_word:
                            found = True
                            break
                        else:
                            continue

                    c = word[idx]
                    if c == '.':
                        for child in cur.children.values():
                            stack.append((child, idx+1))

                    else:
                        if c in cur.children:
                            stack.append((cur.children[c], idx+1))

                return found
        ```
    * Approach2: Recursive:
      * Python Solution
        ```python
        class Node(object):

          def __init__(self):
              self.children = collections.defaultdict(Node)
              self.end_of_word = False

        class WordDictionary:

            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.root = Node()

            def addWord(self, word: str) -> None:
                """
                Adds a word into the data structure.
                """
                cur = self.root

                for c in word:
                    cur = cur.children[c]

                cur.end_of_word = True

            def search(self, word: str) -> bool:
                """
                Returns if the word is in the data structure.
                A word could contain the dot character '.' to represent any one letter.
                """
                def search_with_idx(cur, idx):
                    if idx == w_len:
                        return cur.end_of_word

                    c = word[idx]
                    if c == '.':
                        res = False
                        for child in cur.children.values():
                            if search_with_idx(child, idx+1):
                                res = True
                                break
                        return res
                    else:
                        if c in cur.children:
                            return search_with_idx(cur.children[c], idx+1)
                        return False

                w_len = len(word)
                return search_with_idx(self.root, 0)
        ```
  * 212: Word Search II (H)
  * 642: Design Search Autocomplete System (H)
### BFS & DFS
  * 200: Number of Islands (M)
    * Approach1: BFS, Time: O(mn), Space: O(mn)
      * Set visits to True before append to the queue to **reduce unnecessary iterations.**
      * Python Solution
        ```python
        class NumberOfIsland(object):

          LAND, WATER = '1', '0'
          NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

          @classmethod
          def bfs(cls, grid: List[List[str]]) -> int:
              """
              Time: O(m*n)
              Space: O(m*n)
              """
              def _bfs(r, c):
                if grid[r][c] == cls.WATER or visits[r][c]:
                  return 0

                visits[r][c] = True
                q = collections.deque()
                q.append((r, c))

                while q:
                    r, c = q.popleft()
                    for neigbor in cls.NEIGHBORS:
                        nr, nc = r+neigbor[0], c+neigbor[1]

                        # out of range
                        if nr < 0 or nr >= row or nc < 0 or nc >= col:
                            continue

                        if grid[nr][nc] == cls.WATER or visits[nr][nc]:
                            continue

                        visits[nr][nc] = True
                        q.append((nr, nc))

                return 1

              if not grid or not grid[0]:
                  return 0

              row, col = len(grid), len(grid[0])
              area_cnt = 0
              visits = [[False for _ in range(col)] for _ in range(row)]
              for r in range(row):
                  for c in range(col):
                      area_cnt += _bfs(r, c)
              return area_cnt
        ```
    * Approach2: DFS, Time: O(mn), Space: O(mn)
      * Implementation is just like BFS, but use stack instead
      * Set visits to True before append to the queue to reduce unnecessary iterations.
    * Approach3: DFS recursive,
      * Python Solution
        ```python
        class NumberOfIsland(object):
          LAND, WATER = '1', '0'
          NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

          @classmethod
          def dfs(cls, grid: List[List[str]]) -> int:
              """
              Time: O(m*n)
              Space: O(m*n)
              """
            def _dfs(r, c):
              # check if outouf boundary
              if r < 0 or r >= row or c < 0 or c >= col:
                  return 0

              if grid[r][c] == cls.WATER or visits[r][c]:
                  return 0

              visits[r][c] = True

              for neigbor in cls.NEIGHBORS:
                  _dfs(r+neigbor[0], c+neigbor[1])

              return 1

            if not grid or not grid[0]:
              return 0

            row, col = len(grid), len(grid[0])
            area_cnt = 0
            visits = [[False for _ in range(col)] for _ in range(row)]
            for r in range(row):
                for c in range(col):
                    area_cnt += _dfs(r, c)

            return area_cnt
        ```
    * Approach4: Union Find (Disjoint Set)
  * 339: **Nested List** Weight Sum (E)
    * The weight is defined from top down.
    * n: total number of nested elements
    * d: maximum level of nesting in the input
    * Approach1: BFS, Time: O(n), Space: O(n)
      * Python Solution:
        ```python
        def depthSum(self, nestedList: List[NestedInteger]) -> int:
          if not nestedList:
              return 0

          q = collections.deque()
          for n in nestedList:
              q.append(n)

          depth_sum = 0
          depth_level = 1
          while q:
              q_len = len(q)
              for _ in range(q_len):
                  item = q.popleft()
                  # integer
                  if item.isInteger():
                      depth_sum += (depth_level * item.getInteger())
                  # nested list
                  else:
                      # unpack ths list for next level
                      for n in item:
                          q.append(n_)
              depth_level += 1

          return depth_sum
        ```
    * Approach2: DFS Recursice, Time: O(n), Space: O(n)
      * Python Solution
        ```python
        def depthSum(self, nestedList: List[NestedInteger]) -> int:
          def _dfs(nlist, depth):
              sum_depth = 0
              for n in nlist:
                  if n.isInteger():
                      sum_depth += (n.getInteger() * depth)
                  else:
                      sum_depth += _dfs(n.getList(), depth+1)

              return sum_depth

          return _dfs(nlist=nestedList, depth=1)
        ```
    * Approach3: DFS Iteraitve, Time: O(n), Space: O(d)
      * Python Solution
        ```python
        def depthSum(self, nestedList: List[NestedInteger]) -> int:
          if not nestedList:
              return 0

          sum_depth = 0
          stack = list()
          for n in nestedList:
              stack.append((n, 1))

          while stack:
              item, d = stack.pop()
              if item.isInteger():
                  sum_depth += d * item.getInteger()
              else:
                  for n in item.getList():
                      stack.append((n, d+1))

          return sum_depth
        ```
  * 364: **Nested List** Weight Sum II (M)
    * The weight is defined from bottom up.
    * n: total number of nested elements
    * d: maximum level of nesting in the input
    * Approach1: BFS, Time: O(n), Space: O(n)
      * Without depth variable
      * For example:
        * [1, [2, [3]]]
          * The result is 1 * 3 + 2 * 2 + 3 * 1 = 10
          * iter1:
            * acc = 1
            * depth_sum = 1
          * iter2:
            * acc = 1 + 2
            * depth sum = 1 + 1 + 2
          * iter3:
            * acc = 1 + 2 + 3
            * depth sum = 1 + 2 + 3
      * Python Solution
      ```python
        def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
          if not nestedList:
              return 0

          q = collections.deque()
          depth_sum = 0
          acc = 0

          for n in nestedList:
              q.append(n)

          while q:
              q_len = len(q)
              for _ in range(q_len):
                  item = q.popleft()
                  if item.isInteger():
                      acc += item.getInteger()
                  else:
                      for n in item.getList():
                          q.append(n)
              depth_sum += acc

          return  depth_sum
      ```
    * Approach2: DFS recursive, Time: O(n), Space: O(d)
      * Use hash table to store acc val for each level
      * Post process to get the result
      * Python Solution
        ```python
        def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:

          def _dfs(nestedList, depth):
              for n in nestedList:
                  if n.isInteger():
                      d[depth] += n.getInteger()
                  else:
                      _dfs(n.getList(), depth+1)

        if not nestedList:
            return 0

        d = collections.defaultdict(int)

        _dfs(nestedList, 1)

        # post process to calculate sum_depth
        sum_depth = 0
        max_depth = 1
        for depth in d.keys():
            max_depth = max(max_depth, depth)

        for depth, val in d.items():
            sum_depth += (val *(max_depth-depth+1))

        return sum_depth

        ```
  * 286: Walls and Gates (M)
    * r: number of rows
    * c: number of columns
    * Approach1: BFS, search from **EMPTY**, Time: O(**(rc)^2**), Space: O(rc))
      * This approach can not reuse the calculation info.
      * Time: O(**(rc)^2**)
      * Space: O(rc))
      * Python Solution:
        ```python
        def wallsAndGates(self, rooms: List[List[int]]) -> None:
            """
            Do not return anything, modify rooms in-place instead.
            """
            if not rooms:
                return

            row = len(rooms)
            col = len(rooms[0])

            WALL, GATE, EMPTY = -1, 0, 2147483647
            NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

            def _bfs(r, c):

                visits = [[False for _ in range(col)] for _ in range(row)]

                # row, col, distance
                q = collections.deque()
                visits[r][c] = True
                q.append((r, c))
                distance = 0

                while q:
                    q_len = len(q)
                    for _ in range(q_len):

                        r, c = q.popleft()

                        for neigbor in NEIGHBORS:

                            nr, nc = r+neigbor[0], c+neigbor[1]

                            if nr < 0 or nr >= row or nc < 0 or nc >= col:
                                continue

                            if rooms[nr][nc] == WALL or visits[nr][nc]:
                                continue

                            if rooms[nr][nc] == GATE:
                                return distance + 1

                            visits[nr][nc] = True
                            q.append((nr, nc))

                    distance += 1

                return EMPTY

            for r in range(row):
                for c in range(col):
                    if rooms[r][c] == EMPTY:
                        rooms[r][c] = _bfs(r, c)
        ```
    * Approach2: BFS, search from **GATE**, Time: O((rc)), Space: O(rc))
      * This approach **can avoid recalculation**.
      * Time: O((rc))
      * Space: O(rc))
        * Queue Size
      * Python Solution:
        ```python
        def wallsAndGates(rooms: List[List[int]]) -> None:
            """
            Do not return anything, modify rooms in-place instead.
            """
            if not rooms:
                return

            row = len(rooms)
            col = len(rooms[0])

            WALL, GATE, EMPTY = -1, 0, 2147483647
            NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

            q = collections.deque()
            for r in range(row):
                for c in range(col):
                    # search from the gate
                    if rooms[r][c] == GATE:
                        q.append((r, c))

            distance = 0
            while q:
                q_len = len(q)
                distance += 1
                for _ in range(q_len):
                    r, c = q.popleft()
                    for neigbor in NEIGHBORS:
                        nr, nc = r+neigbor[0], c+neigbor[1]

                        if nr < 0 or nr >= row or nc < 0 or nc >= col:
                            continue

                        if rooms[nr][nc] != EMPTY: # WALL or GATE
                            continue

                        rooms[nr][nc] = distance
                        q.append((nr, nc))
        ```
    * DFS
  * 130: Surrounded Regions (M)
    * A region is captured by flipping all 'O's into 'X's in that surrounded region.
    * BFS: Time:O(rc), Space:O(rc)
      * Try to Group region
      * Python Solution
        ```python
        NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))
        START, END = 'O', 'X'

        class Solution:
            def solve(self, board: List[List[str]]) -> None:
                def _flip_surround_region(r, c):
                    q = collections.deque([(r, c)])
                    flip_regions = [(r, c)]
                    visits[r][c] = True
                    should_flip = True

                    while q:
                        r, c = q.popleft()

                        for neighbor in NEIGHBORS:

                            nr, nc = r + neighbor[0], c + neighbor[1]
                            # check boundary
                            if nr < 0 or nr >= row or nc < 0 or nc >= col:
                                # do not break here since we try to extend the max regino as possible
                                should_flip = False
                                continue

                            # correct border
                            if board[nr][nc] == END or visits[nr][nc]:
                                continue

                            # group region
                            if board[nr][nc] == START:
                                visits[nr][nc] = True
                                flip_regions.append((nr, nc))
                                q.append((nr, nc))

                    if not should_flip:
                        return

                    while flip_regions:
                        r, c = flip_regions.pop()
                        board[r][c] = END

                """
                Do not return anything, modify board in-place instead.
                """
                if not board:
                    return

                row = len(board)
                col = len(board[0])
                visits = [[False for _ in range(col)] for _ in range(row)]

                for r in range(row):
                    for c in range(col):
                        if board[r][c] == START and not visits[r][c]:
                            _flip_surround_region(r, c)
        ```
  * 127: **Word Ladder** (M)
      * Assume length of words is k and number of words is n
      * Approach1: BFS, Time:O(nk), Space:O(n*k^2)
        * Time: O(n*d)
          * Build transformation dict cost: O(n*d)
          * Find the target word in the transformation dict cost: O(n*d)
        * Space: O(n*k^2)
          * Transformatino dict cost: O(n*k^2)
            * Total d transformation for n words, each word cost nd
          * Max queue size: O(n*k)
        * Python Solution
          ```python
          def ladderLength(self, beginWord, endWord, wordList):
            if not wordList:
                return 0

            if beginWord ==  endWord:
                return 1

            '''
            generate word dictionary,
            hit -> *it, h*t, hi*
            '''
            w_combo_d = collections.defaultdict(list)
            for w in wordList:
                for i in range(len(w)):
                    w_key = f'{w[:i]}*{w[i+1:]}'
                    w_combo_d[w_key].append(w)

            q = collections.deque()
            q.append(beginWord)
            visited = {beginWord: True}
            level = 1
            '''
            BFS to find the shorted transformation
            '''
            while q:
                q_len = len(q)
                level += 1
                for _ in range(q_len):
                    w = q.popleft()
                    for i in range(len(w)):
                        # transfrom key
                        w_key = f'{w[:i]}*{w[i+1:]}'

                        if w_key not in w_combo_d:
                            continue

                        for next_transform_w in w_combo_d[w_key]:
                            if next_transform_w in visited:
                                continue

                            visited[next_transform_w] = True

                            if endWord == next_transform_w:
                                return level

                            q.append(next_transform_w)

                        w_combo_d.pop(w_key)
            return 0
          ```
      * Approach2: Bidirectional BFS
  * 126: **Word Ladder** II (H)
  * 079: Word Search (M)
     * L is the word length
     * The key issue is how to reset the used infomation.
     * Approach1: DFS Recursive: Time:O(mn*4^L), Space:O(mn)
       * Time:
         * The first character takes O(mn)
         * Remain characters take O(4^L)
       * Python
         ```python
         def exist(self, board: List[List[str]], word: str) -> bool:

          def dfs(r, c, idx):
              if idx == w_len:
                  return True

              # check boundary
              if not (0 <= r < row and 0 <= c < col):
                  return False

              if visited[r][c] or board[r][c] != word[idx]:
                  return False

              # set
              visited[r][c] = True

              # check next character
              found = False
              for d in directions:
                  found = dfs(r+d[0], c+d[1], idx+1)
                  if found:
                      break

            # reset
            visited[r][c] = False

            return found

          row, col = len(board), len(board[0])

          visited = [[False] * col for _ in range(row)]

          w_len = len(word)

          directions = ((1, 0), (0, 1), (-1, 0), (0, -1))

          found = False
          # find the first character
          for r in range(row):
              for c in range(col):
                  found = dfs(r, c, 0)
                  if found:
                      return found

          return found
         ```
     * Approach2: DFS Iterative: Time:O(mn*4^L), Space:O(mn)
       * Use another stack to track unsed info
       * Python
        ```python
        def exist(self, board: List[List[str]], word: str) -> bool:

          row, col = len(board), len(board[0])
          w_len = len(word)

          directions = ((1, 0), (0, 1), (-1, 0), (0, -1))

          visited = [[False] * col for _ in range(row)]
          visited_track = []
          visited_clean_flag = (-1, -1, -1, True)

          stack = []
          # find the first character
          for r in range(row):
              for c in range(col):
                  stack.append((r, c, 0, False))

          found = False
          while stack:

              r, c, idx, clean_visited = stack.pop()

              if clean_visited:
                  r, c = visited_track.pop()
                  visited[r][c] = False
                  continue

              if idx == w_len:
                  found = True
                  break

              if not (0 <= r < row and 0 <= c < col) \
                  or visited[r][c] or board[r][c] != word[idx]:
                  continue

              visited[r][c] = True
              # help to reset the visited[r][c]
              visited_track.append((r, c))
              stack.append(visited_clean_flag)

              for d in directions:
                  stack.append((r + d[0], c + d[1],
                                idx + 1,
                                False))

          return found
        ```
  * 329: Longest Increasing Path in a Matrix (H)
    * Do not need to keep visitedd ue to the increasing property.
    * Approach1: Recursive, Time:O(2^(m+n)), Space:O(mn)
      * Python
        ```python
        def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
          def dfs(r, c, cur_lip):
              nonlocal lip

              for d in directions:
                  rc, cc = r + d[0], c + d[1]
                  if 0 <= rc < row and 0 <= cc < col \
                      and matrix[rc][cc]  > matrix[r][c]:
                          lip = max(lip, cur_lip + 1)
                          dfs(rc, cc, cur_lip + 1)

          row, col = len(matrix), len(matrix[0])
          directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
          # longest increasing path
          lip = 0
          for r in range(row):
              for c in range(col):
                  dfs(r, c, 1)

          return lip
        ```
    * Approach2: Recursive, Time:O((mn)), Space:O(mn)
      * Python
        ```python
        def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
          def dfs(r, c):
              if memo[r][c]:
                  return memo[r][c]

              # the base case is 1 and without increasing neighbors.
              memo[r][c] = 1

              for d in directions:
                  rc, cc = r + d[0], c + d[1]
                  if 0 <= rc < row and 0 <= cc < col \
                      and matrix[rc][cc] > matrix[r][c]:
                          memo[r][c] = max(memo[r][c], dfs(rc, cc) + 1)

              return memo[r][c]

          if not matrix:
              return 0

          row, col = len(matrix), len(matrix[0])

          if not col:
              return 0

          directions = ((1, 0), (0, 1), (-1, 0), (0, -1))

          memo = [[None for _ in range(col)] for _ in range(row)]

          # longest increasing path
          lip = 0
          for r in range(row):
              for c in range(col):
                  lip = max(lip, dfs(r, c))

          return lip
        ```
    * Approach3: Topological Sort, Time:O((mn)), Space:O(mn)
      * Definition:
        * If matrix[i][j] < matrix[k][l], path: matrix[i][j] -> matrix[k][l]
        * indegree:
          * (i, j): 0
          * (k, l): 1
        * outdegree:
          * (i, j): [(k, l)]
          * (k, l): []
          * do not need to keep, we can get from neighbros of matrix
      * Starting from the nodes with indegree == 0 ( max val in the area)
      * Python
        ```python
        def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
          if not matrix:
              return 0

          row, col = len(matrix), len(matrix[0])

          if not col:
              return 0

          directions = ((1, 0), (0, 1), (-1, 0), (0, -1))

          indegree = [[0 for _ in range(col)] for _ in range(row)]
          for r in range(row):
              for c in range(col):
                  for d in directions:
                      rc, cc = r + d[0], c + d[1]
                      if 0 <= rc < row and 0 <= cc < col \
                          and matrix[r][c] < matrix[rc][cc]:
                          indegree[rc][cc] += 1

          q = collections.deque()
          for r in range(row):
              for c in range(col):
                  if indegree[r][c] == 0:
                      q.append((r,c))

          # longest increasing path
          lip = 0
          while q:
              q_len = len(q)
              lip += 1
              for _ in range(q_len):
                  r, c = q.popleft()
                  for d in directions:
                      rc, cc = r + d[0], c + d[1]
                      if 0 <= rc < row and 0 <= cc < col \
                          and matrix[r][c] < matrix[rc][cc]:
                              indegree[rc][cc] -= 1
                              if indegree[rc][cc] == 0:
                                  q.append((rc,cc))

          return lip
        ```
  * 051: N-Queens (H)
  * 052: N-Queens II (H)
### Dynamic Programming
  * Ref:
    * [From good to great. How to approach most of DP problems](https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.)
  * This particular problem and most of others can be approached using the following sequence:
    * Recursive relation
    * Recursive (top-down)
    * Recursive + memo (top-down)
    * Iterative + memo (bottom-up)
    * Iterative + N variables (bottom-up)
  * Fibonacci sequence:
    * 509: Fibonacci Number (E)
      * Recursive relation
        * n == 0
        * n == 1
        * n > 1
          * f(n) = f(n-1) + f(n-2)
      * Approach1: Recursive + memo (top-down), Time:O(n), Space:(n)
        * Python Solution
        ```python
        def fib(self, N: int) -> int:
          def _fib(n):
              if n <= 1:
                  return memo[n]

              if not memo[n]:
                  memo[n] = _fib(n-1) + _fib(n-2)

              return memo[n]

          if N == 0:
              return 0
          if N == 1:
              return 1

          memo = [None] * (N+1)
          memo[0] = 0
          memo[1] = 1

          return _fib(N)
        ```
      * Approach2: Iterative + memo (bottom-up), Time:O(n), Space:(n)
        * Python Solution
        ```python
        def fib(self, N: int) -> int:
          if N == 0:
              return 0
          if N == 1:
              return 1

          memo = [None] * (N+1)
          memo[0] = 0
          memo[1] = 1

          for i in range(2, N+1):
              memo[i] = memo[i-1] + memo[i-2]

          return memo[N]
        ```
      * Approach3: Iterative + N variables (bottom-up), Time:O(n), Space:(1)
        * Python Solution
        ```python
        def fib(self, N: int) -> int:
          if N == 0:
              return 0
          if N == 1:
              return 1

          prev = 0
          cur = 1
          for _ in range(2, N+1):
              prev, cur = cur, prev+cur

          return cur
        ```
    * 070: Climbing Stairs (E)
      * Recursive relation
        * n <= 2:
          * n == 0 or n == 1
            * 1
          * n == 2
            * 2 (1 + 1 or 2)
        * n > 2:
          * f(n) = f(n-2) + f(n-1)
            * f(n-2) + 2 step
            * f(n-1) + 1 step
      * Approach1: Recursive + memo (top-down), Time:O(n), Space:O(n)
      * Approach2: Iterative + memo (bottom-up), Time:O(n), Space:O(n)
        * Python Solutino
        ```python
        def climbStairs(self, n: int) -> int:
          """
          Each time you can either climb 1 or 2 steps.
          In how many distinct ways can you climb to the top?
          """
          if n == 0 or n == 1:
              return 1

          if n == 2:
              return 2

          # n >= 3
          memo = [None] * (n + 1)
          memo[0] = memo[1]= 1
          memo[2] = 2

          # 3 ~ n
          for i in range(3, n+1):
              memo[i] = memo[i-1] + memo[i-2]

          return memo[n]
        ```
      * Approach3: Iterative + N variables (bottom-up), Time:O(n), Space:O(1)
        * Python Solution
        ```python
        def climbStairs(self, n: int) -> int:
          """
          Each time you can either climb 1 or 2 steps.
          In how many distinct ways can you climb to the top?
          """
          if n == 0 or n == 1:
              return 1

          if n == 2:
              return 2

          prev = 1
          cur = 2
          for _ in range(3, n+1):
              prev, cur = cur, prev + cur

          return cur
        ```
      * Approach4: [Log(n) Solution](https://leetcode.com/problems/climbing-stairs/solution/)
    * 091: **Decode Ways** (M)
      * Recursive relation
        * Check conditions
          * one character
            ```python
            return True if '1' <= c <= '9' else False
            ```
          * two characters
            ```python
            return True if '10' <= cc <= '26' else False
            ```
        * Conditional Fibonaccci number
          * n == 0
            * check s[0]
          * n == 1
            * check s[0] + s[0] and s[0:2]
          * n >= 2:
            * init
              * f(n) = 0
            * if check s[n] == True
              * f(n) += f(n-1)
            * if check s[n-1:n+1] == True
              * f(n) += f(n-2)
      * Approach1: Iterative + memo (bottom-up), Time:O(n), Space:O(n)
        * Python Solution
          ```python
          def numDecodings(self, s: str) -> int:
            """
            'A' -> 1
            'B' -> 2
            ...
            'Z' -> 26
            Given a non-empty string containing only digits
            determine the total number of ways to decode it.
            """
            def check_s_char(c):
                return True if '1' <= c <= '9' else False

            def check_d_char(cc):
                return True if '10' <= cc <= '26' else False

            n = len(s)
            memo = [0] * n

            if n == 0:
                return 0

            if check_s_char(s[0]):
                memo[0] = 1
            else:
                memo[0] = 0
            if n == 1:
                return memo[0]

            if memo[0] and check_s_char(s[1]):
                memo[1] += 1
            if check_d_char(s[0:2]):
                memo[1] += 1

            for i in range(2, n):
                if check_s_char(s[i]):
                    memo[i] += memo[i-1]

                if check_d_char(s[i-1:i+1]):
                    memo[i] += memo[i-2]

            return memo[n-1]
          ```
      * Approach2: Iterative + N variables (bottom-up), Time:O(n), Space:O(1)
        * Python Solution
        ```python
        def numDecodings(self, s: str) -> int:
            """
            'A' -> 1
            'B' -> 2
            ...
            'Z' -> 26
            Given a non-empty string containing only digits
            determine the total number of ways to decode it.
            """
            def check_s_char(c):
                return True if '1' <= c <= '9' else False

            def check_d_char(cc):
                return True if '10' <= cc <= '26' else False

            n = len(s)
            memo = [0] * n

            if n == 0:
                return 0

            prev = cur = 0
            if check_s_char(s[0]):
                prev = 1
            if n == 1:
                return prev

            if prev and check_s_char(s[1]):
                cur += 1
            if check_d_char(s[0:2]):
                cur += 1

            for i in range(2, n):
                last_cur = cur
                cur = 0
                if check_s_char(s[i]):
                    cur += last_cur

                if check_d_char(s[i-1:i+1]):
                    cur += prev

                prev = last_cur

            return cur
        ```
  * **One Dimensional**:
    * 062: Unique Paths (M)
      * Approach1: Combination:
        * total m-1 down steps and n-1 right steps
        * (m+n)!/m!n!
        * Python Solution
        ```python
        from math import factorial as f
        def uniquePaths(self, m: int, n: int) -> int:
          if not m and not n:
              return 0

          if m == 1 or n == 1:
              return 1

          return int(f(m+n-2) /(f(m-1) * f(n-1)))
        ```
      * Approach2: DP, Time:O(mn), Space:O(mn)
        * Recursive relation
        * for r == 0
          * f(0, c) = 1
        * for c == 0
          * f(r, 0) = 1
        * For r > 0 and c > 0
          * f(r,c) = f(r-1) + f(c-1)
        * Python Solution
          ```python
          def uniquePaths(self, m: int, n: int) -> int:
            if not m and not n:
                return 0

            if m == 1 or n == 1:
                return 1

            memo = [[0 for _ in range(n)] for _ in range(m)]

            # init the first row
            for c in range(n):
                memo[0][c] = 1

            # init the first column
            for r in range(m):
                memo[r][0] = 1

            for r in range(1, m):
                for c in range(1, n):
                    memo[r][c] = memo[r-1][c] + memo[r][c-1]

            return memo[m-1][n-1]
          ```
      * Approach3: DP, Time:O(mn), Space:O(n)
        * In fact, keep one row is enough
        * Python Solution
        ```python
        def uniquePaths(self, m: int, n: int) -> int:
          if not m and not n:
              return 0

          if m == 1 or n == 1:
              return 1

          # init first row
          memo = [1 for _ in range(n)]

          # 1 to m-1 rows
          for _ in range(1, m):
              for c in range(1, n):
                  # memo[c] = memo[c] (from up) + memo[c-1] (from left)
                  memo[c] += memo[c-1]

          return memo[n-1]
        ```
    * 063: Unique Paths II (M)
    * 120: Triangle (M)
    * 279: Perfect Squares (M)
    * 139: Word Break (M)
    * 375: Guess Number Higher or Lower II (M)
    * 322: Coin Change (H)
    * 312: Burst Balloons (H)
  * **Two Dimensional**:
    * Longest Common Subsequence:
      * Ref:
         * https://www.youtube.com/watch?v=NnD96abizww
      * Approach1: DP: Time: O(n^2), Space: O(n^2)
        * Python Solution
         ```python
          @staticmethod
          def traverse_lcs_memo(s1, s1_idx, s2, s2_idx, memo):
              """ traverse the memo array to find lcs
              """
              lcs_len = memo[s1_idx][s2_idx]
              lcs = [None] * lcs_len
              i, j = s1_idx, s2_idx

              while lcs_len:
                  if s1[i] == s2[j]:
                      lcs_len -= 1
                      lcs[lcs_len] = s1[i]
                      i -= 1
                      j -= 1
                  else:
                      if memo[i][j] == memo[i-1][j]:
                          i -= 1
                      else:  # memo[i][j] == memo[i][j-1]
                          j -= 1

              return "".join(lcs)

          @staticmethod
          def lcs(s1, s2):
              """
              Longest common sequence
              Time: O(n^2), Space: O(n^2)
              Ref: https://www.youtube.com/watch?v=NnD96abizww
              """
              if not s1 or not s2:
                  return ""

              l1, l2 = len(s1), len(s2)

              memo = [[0 for _ in range(l2)] for _ in range(l1)]

              if s1[0] == s2[0]:
                  memo[0][0] = 1

              # init first row
              for j in range(1, l2):
                  if memo[0][j-1] or s2[j] == s1[0]:
                      memo[0][j] = 1

              # init first column
              for i in range(1, l1):
                  if memo[i-1][0] or s1[i] == s2[0]:
                      memo[i][0] = 1

              # complete the memo
              for i in range(1, l1):
                  for j in range(1, l2):
                      if s1[i] == s2[j]:
                          memo[i][j] = memo[i-1][j-1] + 1
                      else:
                          memo[i][j] = max(memo[i-1][j], memo[i][j-1])

              return StrUtils.traverse_lcs_memo(s1, l1-1,
                                                s2, l2-1,
                                                memo)
         ```
    * 256: Paint House (E)
      * There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.
      * Recursive relation
        * n == 0: (first house)
          * f_color[0][red] = costs[0][red]
          * f_color[0][blue] = costs[0][green]
          * f_color[0][green] = costs[0][green]
        * n > 0:
          * For each color at nth house, we have 3 choices
            * f_color[n][red] = costs[n][red] + min(f_color[n-1][blue], f_color[n-1][green])
            * f_color[n][blue] = costs[n][blue] + min(f_color[n-1][red], f_color[n-1][green])
          * f_  color[n][green] = costs[n][green] + min(f_color[n-1][red], f_color[n-1][blue])
        * and the minimum price if we painted to nth house would be
          * min(f_color[n][red], f_color[n][blue], f_color[n][green])
      * Approach1: Iterative + memo (bottom-up), Time:O(nk), Space:(nk)
        * Python Solution
          ```python
          def minCost(self, costs: List[List[int]]) -> int:
            """
            paint all the houses such that no two adjacent houses have the same color
            Find the minimum cost to paint all houses
            cost: n x 3 matrix for each house
            red, blue, green
            """
            if not costs:
                return 0

            house_cnt = len(costs)
            acc_cost = [[0 for _ in range(3)] for _ in range(house_cnt)]

            r_idx, b_idx, g_idx = 0, 1, 2

            # For the first house
            acc_cost[0] = costs[0]

            # For 1th to (n-1)th house
            for i in range(1, house_cnt):
                acc_cost[i][r_idx] = costs[i][r_idx] + min(acc_cost[i-1][b_idx], acc_cost[i-1][g_idx])
                acc_cost[i][b_idx] = costs[i][b_idx] + min(acc_cost[i-1][r_idx], acc_cost[i-1][g_idx])
                acc_cost[i][g_idx] = costs[i][g_idx] + min(acc_cost[i-1][r_idx], acc_cost[i-1][b_idx])

            # unpack the last row
            return min(*acc_cost[house_cnt-1])
          ```
      * Approach2: Iterative + N variables (bottom-up), Time:O(nk), Space:(n)
        * Python Solution:
          ```python
          def minCost(self, costs: List[List[int]]) -> int:
            """
            paint all the houses such that no two adjacent houses have the same color
            Find the minimum cost to paint all houses
            cost: n x 3 matrix for each house
            red, blue, green
            """
            if not costs:
                return 0

            house_cnt = len(costs)
            r_idx, b_idx, g_idx = 0, 1, 2

            # For the 0th house
            cur_r, cur_b, cur_g = costs[0][r_idx], costs[0][b_idx], costs[0][g_idx]

            # For 1th to (n-1)th house
            for i in range(1, house_cnt):
                last_r, last_b, last_g = cur_r, cur_b, cur_g
                cur_r = costs[i][r_idx] + min(last_b, last_g)
                cur_b = costs[i][b_idx] + min(last_r, last_g)
                cur_g = costs[i][g_idx] + min(last_r, last_b)

            return min(cur_r, cur_b, cur_g)
            ```
    * 265: Paint House II (H) (L)
    * 064: Minimum Path Sum (M)
    * 072: Edit Distance (H)
      * Ref:
        * https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and-intuition
      * Recursive relation
        * if word1[i] == word[j]  # no edit
          * memo[i][j] = memo[i-1][j-1]
        * elif word1[i] =! word[j]:
          * memo[i][j] = minimum of
            * 1 + memo[i][j-1]    # insert
            * 1 + memo[i-1][j]    # delete
            * 1 + memo[i-1][j-1]  # replacement
      * Recursive (top-down)
        * Python Solution
        ```python
          def minDistance(self, word1: str, word2: str) -> int:
            def min_distance_idx(idx1, idx2):
                # run out both string
                if idx1 == -1 and idx2 == -1:
                    return 0

                if idx1 == -1:
                    return idx2 + 1

                if idx2 == -1:
                    return idx1 + 1

                if word1[idx1] == word2[idx2]:
                    return  min_distance_idx(idx1-1, idx2-1)
                else:
                    replace = min_distance_idx(idx1-1, idx2-1)
                    insert = min_distance_idx(idx1, idx2-1)
                    delete = min_distance_idx(idx1-1, idx2)
                    return 1 + min(replace, insert, delete)

              return min_distance_idx(len(word1)-1, len(word2)-1)
        ```
      * Iterative + memo (bottom-up)
        * Python Solution
          ```python
          def minDistance(self, word1: str, word2: str) -> int:

            if not word1 and not word2:
                return 0

            if not word1:
                return len(word2)

            if not word2:
                return len(word1)

            n1, n2 = len(word1), len(word2)
            row, col = n1 + 1, n2 + 1

            memo = [[None for _ in range(col)] for _ in range(row)]

            # init first row,
            # assume there is a dummy character before word1 and word2
            for c in range(col):
                # insert operation
                memo[0][c] = c

            # init first col
            for r in range(row):
                # insert operation
                memo[r][0] = r

            for i in range(1, row):
                for j in range(1, col):
                    if word1[i-1] == word2[j-1]:
                        memo[i][j] = memo[i-1][j-1]
                    else:
                        memo[i][j] = 1 + min(memo[i-1][j-1],
                                             memo[i][j-1],
                                             memo[i-1][j])

            return memo[row-1][col-1]
          ```
    * 097: Interleaving String (H)
    * 174: Dungeon Game (H)
    * 221: Maximal Square (M)
    * 085: Maximal Rectangle (H)
    * 363: Max Sum of Rectangle No Larger Than K	TreeSe (H)
  * **Deduction**:
    * 714: Best Time to Buy and Sell Stock with Transaction Fee (M) (Array)
    * 276: Paint Fence (E)
      * There is a fence with n posts, each post can be painted with one of the k colors.
      * You have to paint all the posts such that no more than two adjacent fence posts have the same color.
      * Recursive relation
        * n <= 2:
          * n == 0:
            * 0 color
          * n == 1:
            * k color
          * n == 2:
            * f_same: k * 1
            * f_diff: k * (k-1)
            * return f_same + f_diff
        * n > 2
          * f_same(i) = f_diff(i-1) * 1
          * f_diff(i) = (f_same(i-1) + f_diff(i-1)) * (k-1)
          * return f_same(i) + f_diff(i)
      * Approach1: Recursive + memo (top-down)
      * Approach2: Iterative + memo (bottom-up), Time: O(n), Space: O(n)
        * Python Solution
          ```python
          def numWays(self, n: int, k: int) -> int:
            """
            n: number of posts
            k: number of color
            """
            # no posts
            if n == 0:
                return 0

            # 1 posts
            if n == 1:
                return k

            # 2 posts
            memo_same = [None] * n
            memo_same[1] = k

            memo_diff = [None] * n
            memo_diff[1] = k * (k - 1)

            # 2 to n-1
            for i in range(2, n):
                memo_same[i] = memo_diff[i-1]
                memo_diff[i] = (memo_diff[i-1] +memo_same[i-1]) * (k - 1)

            return memo_same[n-1] + memo_diff[n-1]
          ```
      * Approach3: Iterative + N variables (bottom-up), Time: O(n), Space: O(1)
        * Python Solution
          ```python
          def numWays(self, n: int, k: int) -> int:
            """
            n: number of posts
            k: number of color
            """
            if n == 0:
                return 0

            if n == 1:
                return k

            # post == 2
            same = k
            diff = k * (k-1)

            # for adding post==3 to post==n
            for _ in range(3, n+1):
                tmp = diff

                # diff[i] = same[i-1] * (k-1) + diff[i-1] * (k-1)
                diff = (diff + same) * (k-1)

                # same[i] = diff[i-1] * 1
                same = tmp

            return diff + same
    * 198: House Robber (E)
      * Recursive relation
        * For n < 2
          * f(0) = nums[0]
          * f(1) = max(nums[0], nums[1])
        * For n >= 2
          * f(n) = max(f(n-2)+monery(n), F(n-1))
      * Recursive + memo (top-down), Time:O(n), Space:(n)
        * Python Solution
          ```python
          def rob(self, nums: List[int]) -> int:
          """
          It will automatically contact the police if two adjacent houses were broken
          into on the same night.
          """
            # R(n) = max(R(n-2)+monery(n), R(n-1))
            def _rob(n):
                if n == 0:
                    return memo[0]
                elif n == 1:
                    return memo[1]
                else:  # n >= 2
                    if not memo[n]:
                        memo[n] = max(_rob(n-2)+nums[n], _rob(n-1))
                    return memo[n]

            if not nums:
                return 0

            if len(nums) == 1:
                return nums[0]

            memo = [None] * len(nums)
            memo[0] = nums[0]               # 1 house
            memo[1] = max(nums[0], nums[1]) # 2 houses
            return _rob(len(nums)-1)
          ```
      * Iterative + memo (bottom-up), Time:O(n), Space:(1)
        * Python Solution
        ```python
        def rob(self, nums: List[int]) -> int:
          """
          It will automatically contact the police if two adjacent houses were broken
          into on the same night.
          """
          if not nums:
              return 0

          if len(nums) == 1:
              return nums[0]

          memo = [None] * len(nums)
          memo[0] = nums[0]               # 1 house
          memo[1] = max(nums[0], nums[1]) # 2 houses

          for i in range(2, len(nums)):
              memo[i] = max(memo[i-2]+nums[i], memo[i-1])

          return memo[len(nums)-1]
        ```
      * Iterative + N variables (bottom-up), Time:O(n), Space:(1)
        * Python Solution
        ```python
        def rob(self, nums: List[int]) -> int:
        """
        It will automatically contact the police if two adjacent houses were broken
        into on the same night.
        """
        if not nums:
            return 0

        if len(nums) == 1:
            return nums[0]

        prev_max = nums[0]
        cur_max = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            prev_max, cur_max = cur_max, max(prev_max+nums[i], cur_max)

        return cur_max
        ```
    * 213: House Robber II (M)
      * All houses at this place are arranged **in a circle**.
      * How to break the circle ?
        * (1) i not robbed
        * (2) i robbed
        * (3) pick the bigger one.
      * Python
        ```python
        def rob(snums: List[int]) -> int:
            def rob_non_cycle(start, end):
                n = end - start + 1
                if n == 1:
                    return nums[start]
                if n == 2:
                    return max(nums[start], nums[start+1])

                prev_max = nums[start]
                cur_max = max(nums[start], nums[start+1])
                for i in range(start+2, end+1):
                    prev_max, cur_max = cur_max, max(nums[i]+prev_max, cur_max)

                return cur_max

            n = len(nums)
            if n == 0:
                return 0
            if n == 1:
                return nums[0]
            if n == 2:
                return max(nums[0], nums[1])
            if n == 3:
                return max(nums[0], nums[1], nums[2])

            # rob the 0th house and skip 1th hourse and (n-1)th house
            rob_first = nums[0] + rob_non_cycle(2, n-2)
            # do not rob the 0th
            do_not_rob_first = rob_non_cycle(1, n-1)

            return max(rob_first, do_not_rob_first)`
        ```
    * 337: House Robber III (M)
      * Ref:
        * https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem
      * Recursive Relation1:
        * terminate case: when tree is empty, return 0
        * case1: Rob node: f(n) = node.val
                                 + f(n.left.left) + f(n.left.right)
                                 + f(n.right.left) + f(n.right.right)
        * case2: Do not rob node: f(n) = f(n.left) + f(n.right)
      * Approach1: Recursive + memo (top-down), Time:O(n), Space:O(n)
        * Python,
          ```python
          def rob(self, root: TreeNode) -> int:
            def dfs(node):
                if not node:
                    return 0

                if node not in memo:
                    rob = node.val
                    if node.left:
                        rob += (dfs(node.left.left) + dfs(node.left.right))
                    if node.right:
                        rob += (dfs(node.right.left) + dfs(node.right.right))

                    do_not_rob = dfs(node.left)+dfs(node.right)

                    memo[node] = max(rob, do_not_rob)

                return memo[node]

            memo = dict()
            return dfs(root)
          ```
      * Recursive Relation2:
          * terminate case: when tree is empty, return 0, 0 (rob and not rob)
          * case1: Rob node: f(n) = node.val + f(n.left)[left_not_rob] + f(n.right)[right_not_rob]
          * case2: Do not rob node: f(n) = max(f(n.left)) + max(f(n.right))
      * Approach2: Recursive without memo (top-down), Time:O(n), Space:O(n)
        * Every node return two values, rob the node and do not rob the node
        * Python
          ```python
          def rob(self, root: TreeNode) -> int:
            def dfs(node):
                if not node:
                    return 0, 0

                l_rob, l_not_rob = dfs(node.left)
                r_rob, r_not_rob = dfs(node.right)

                rob = node.val + l_not_rob + r_not_rob
                do_not_rob = max(l_rob, l_not_rob) + max(r_rob, r_not_rob)

                return rob, do_not_rob

            return max(dfs(root))
          ```
      * Approach3: Iterative, postorder, Time:O(n), Space:O(n)
        * Python
          ```python
          def rob(self, root: TreeNode) -> int:
            if not root:
                return 0

            stack = [root]
            post_order = collections.deque()
            while stack:
                node = stack.pop()
                post_order.appendleft(node)

                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            memo = dict()
            for node in post_order:
                l_rob = l_not_rob = 0
                if node.left:
                    l_rob, l_not_rob = memo[node.left]

                r_rob = r_not_rob = 0
                if node.right:
                    r_rob, r_not_rob = memo[node.right]

                rob = node.val + l_not_rob + r_not_rob
                not_rob = max(l_rob, l_not_rob) + max(r_rob, r_not_rob)

                memo[node] = (rob, not_rob)

            return max(memo[root])
          ```
    * 010: **Regular Expression** Matching (H)
    * 044: **Wildcard** Matching (H)
### Backtracking
  * 294: Flip Game II (M)
  * 022: Generate Parentheses (M)
     * Approach1-1: Brute Force, Recursive, Time:O(n*n^(2n))
       * f(n) = f(n-1) + '('  or f(n-1) + ')'
       * Time Complexity:
         * total n^(2n) combination
       * Space Complexity:
       * Python Solution:
        ```python
        def generateParenthesis(self, n: int) -> List[str]:
          def valid(cur):
              bal = 0
              for bracket in cur:
                  if bracket == '(':
                      bal += 1
                  else:
                      bal -= 1
                      if bal < 0:
                          return False
              return bal == 0

          def backtrack(cur):
              if len(cur) == 2*n:
                  if valid(cur):
                      res.append("".join(cur))
                  return

              # case1: f(n) = f(n-1) + '('
              cur.append('(')
              backtrack(cur)
              cur.pop()
              # case2 : f(n) = f(n-1) + ')'
              cur.append(')')
              backtrack(cur)
              cur.pop()

          res = []
          cur = []
          backtrack(cur)
          return res
        ```
     * Approach1-2: Brute Force, Iterative: Time:O(n*n^(2n))
       * Python Solution
       ```python
       def generateParenthesis(self, n: int) -> List[str]:
          def valid(cur):
              bal = 0
              for bracket in cur:
                  if bracket == '(':
                      bal += 1
                  else:
                      bal -= 1
                      if bal < 0:
                          return False

              return bal == 0

          combos = [[]]
          res = []
          for _ in range(2*n):
              combo_len = len(combos)
              for j in range(combo_len):
                  combo = combos[j]
                  copy=combo.copy()
                  copy.append('(')
                  combos.append(copy)
                  combo.append(')')

          for combo in combos:
              if valid(combo):
                  res.append("".join(combo))

          return res
       ```
     * Approach2-1: Control the left and right, Recursive:
       * Add them only when we know it will remain a valid sequence (cut off the tree)
       * Python Solution:
        ```python
        def generateParenthesis(self, n: int) -> List[str]:
          def backtrack(cur, left, right):
              if left == right == n:
                  partheneses.append(''.join(cur))
                  return

              if left < n:
                  cur.append('(')
                  backtrack(cur, left+1, right)
                  cur.pop()

              if right < left:
                  cur.append(')')
                  backtrack(cur, left, right+1)
                  cur.pop()

          cur = []
          partheneses = []
          backtrack(cur, left=0, right=0)

          return partheneses
        ```
     * Approach2-2: Control the left and right, Iterative:
       * Python Solution:
        ```python
        def generateParenthesis(self, n: int) -> List[str]:
          res = []
          stack = []
          # s, left ,right
          stack.append(('(', 1, 0))
          while stack:
              s, left, right = stack.pop()
              if left == right == n:
                  res.append(s)
                  continue

              if left < n:
                  stack.append((s+'(', left+1, right))

              if right < left:
                  stack.append((s+')', left, right+1))

          return res
        ```
  * **Subset**
    * 078: Subsets (M)
      * Ref:
        * [C++ Recursive/Iterative/Bit-Manipulation](https://leetcode.com/problems/subsets/discuss/27278/C%2B%2B-RecursiveIterativeBit-Manipulation)
      * Approach1: Recursive, Time: O(n*2^n), Space:(2^n):
        * DFS Traverse
        * Example:
          ```txt
          [1,2,3]
          [] -> [1] -> [1,2] -> [1,2,3] (backtrack)
                    -> [1,3] (backtrack)

             -> [2] -> [2,3] (backtrack)

             -> [3] (backtrack)
          ```
          * []
        * Time: O(n*2^n)
          * total 2^n subset, each subset need O(n) to copy
        * Space: O(2^n)
          * Total (2^n) subsets
        * Python
          ```python
          def subsets_rec(nums: List[int]) -> List[List[int]]:
              def _subsets(cur: list, start):
                # make a copy, append subs cur len(nums)-1
                subs.append(cur[:])

                if start == len(nums):
                    return

                for i in range(start, len(nums)):
                    cur.append(nums[i])
                    # !!! start = i =1 rather than start + 1
                    _subsets(cur, start=i+1)
                    cur.pop()

              subs = []
              cur = []
              if not nums:
                  return []

              _subsets(cur, 0)
              return subs
          ```
      * Approach2: Iterative, Time: O(n*2^n), Space:(2^n)
        * Time: O(n*2^n)
          * total:1 + 2 + 4 + 8 ...2^n = O(2^n) round
            * in each round needs O(n) to copy list
        * Space: O(2^n)
          * Total (2^n) subsets (only two status for each character)
        * example: [a, b, c]
          * s1: [[]]
          * s2: [[], [a]] (with a, without a)
          * s3: [[], [a], [b], [a, b]] (with b, without b)
          * s4: [[], [a], [b], [a, b], [c], [a, c], [b, c], [a, b, c]]
        * Python1
          ```python
          def subsets(self, nums: List[int]) -> List[List[int]]:
            subs = [[]]
            for n in nums:
                # copy and append n
                cur = [sub + [n] for sub in subs]
                subs.extend(cur)

            return subs
          ```
        * Python2
          ```python
          def subsets(self, nums: List[int]) -> List[List[int]]:
            subs = [[]]

            for n in nums:
                sub_len = len(subs)
                for i in range(sub_len):
                    sub = subs[i]
                    subs.append(sub[:])
                    sub.append(n)

            return subs
          ```
      * Approach3: Iterative, **bit manipulation**, Time: O(n*2^n), Space:(2^n)
        * Time: O(n*2^n)
          * Total (2^n) round
            * For each round need to populate the new set taking O(n) time
        * Space: O(2^n)
          * Total (2^n) subsets (only two status for each character)
        * example:
            ```txt
            [a, b, c]
            set 000: []
            set 001: [a]
            set 010: [b]
            set 011: [a, b]
            set 100: [c]
            set 101: [a, c]
            set 110: [b, c]
            set 111  [a, b, c]
            ```
        * Python
            ```python
            def subsets_iter_bit(nums: List[int]) -> List[List[int]]:
                subs = []
                subs_len = 2 ** len(nums)
                num_len = len(nums)

                for sub_idx in range(subs_len):
                    new_sub = []
                    for num_idx in range(num_len):
                        if sub_idx & (1 << num_idx):
                            new_sub.append(nums[num_idx])
                    subs.append(new_sub)

                return subs
            ```
    * 090: Subsets II (M)
      * Given a collection of integers that might **contain duplicates**, nums, return all possible subsets (the power set).
      * Approach1: Recursive
        * Python
          ```python
          def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
            def backtrack(start, cur):
                subs.append(cur[:])

                if start == n:
                    return

                for i in range(start, n):
                    # skip duplicate in the same position
                    if i > start and nums[i] == nums[i-1]:
                        continue

                    num = nums[i]
                    cur.append(num)
                    backtrack(i+1, cur)
                    cur.pop()

            if not nums:
                return []

            n = len(nums)
            nums.sort()
            subs = []
            cur = []
            backtrack(0, cur)
            return subs
          ```
      * Approach2: Iterative
        * Ref:
          * https://leetcode.com/problems/subsets-ii/discuss/30166/Simple-python-solution-without-extra-space.
        * Example:
          ```txt
          [1, 2, 2]

          init:
            subs = [[]],
            cur = []

          step1:
            subs = [[], [1]],
            cur = [[1]]

          step2:
            subs = [[], [1], [2], [1,2]]
            cur = [[2], [1,2]]

          step3: nums[i] == nums[i-1]
            subs = [[], [1], [2], [1,2], [2,2], [1,2,2]]
            cur = [[2,2], [1,2,2]]  # iterative from cur in the step2
          ```
        * Python
          ```python
          def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
            if not nums:
                return []

            nums.sort()
            subs = [[]]
            cur = []

            for i in range(len(nums)):
                n = nums[i]

                # same idea, we can not insert duplicate val in the same position
                if i > 0 and nums[i] == nums[i-1]:
                    cur = [sub + [n] for sub in cur]
                else:
                    cur = [sub + [n] for sub in subs]

                subs.extend(cur)

            return subs
          ```
  * **Combinations**
    * 077: Combinations (M)
      * Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
      * Example:
        ```txt
          n = 5, k = 3

          [1] -> [1,2] -> [1,2,3] (len == k, backtrack)
                       -> [1,2,4] (len == k, backtrack)
                       -> [1,2,5] (len == k, backtrack)
                       -> 6 > n, backtrack

              -> [1,3] -> [1,3,4] (len == k, backtrack)
                       -> [1,3,5] (len == k, backtrack)
                       -> 6 > n, backtrack

              -> [1,4] -> [1,4,5] (len == k, backtrack)
                       -> 6 > n, backtrack

              -> [1,5] -> not enough elements, backtrack
                       ->  6 > n, backtrack

          [2] -> [2,3] -> [2,3,4] (len == k, backtrack)
                       -> [2,3,5] (len == k, backtrack)
                       -> 6 > n, backtrack

              -> [2,4] -> [2,4,5] (len == k, backtrack)
                       -> 6 > n, backtrack

              -> [2,5] -> not enough elements, backtrack
          ...
        ```
      * Approach1: Recursive Time: O(k * n!/(n!*(n-k))!), Space: O(n!/(n!*(n-k)!)
        * Time: O(k* n!/(n!*(n-k)!)
          * total n!/(n!*(n-k)! combinations
          * k is the time to pupulate each combinations
        * Space: O(n!/(n!*(n-k)!)
          * Total O(n!/(n!*(n-k)!) combintations
        * Python
            ```python
            def combine(self, n: int, k: int) -> List[List[int]]:
              def backtrack(start, cur):
                  if len(cur) == k:
                      comb.append(cur[:])
                      return

                  # skip the cases that can not satisfy k == len(cur) in the future
                  if k - len(cur) > n - start + 1:
                      return

                  for i in range(start, n+1):
                      cur.append(i)
                      backtrack(i+1, cur)
                      cur.pop()

              if k > n:
                  return []

              comb = []
              cur = []
              backtrack(1, cur)
              return comb
            ```
      * Approach2: Iterative
        * The same idea like recursive approach, use stack to control backtrack
        * Python
          ```python
          def combin_iter_v2(n: int, k: int) -> List[List[int]]:
            """
            Ref: https://leetcode.com/problems/combinations/discuss/27029/AC-Python-backtracking-iterative-solution-60-ms
            Time: O(k* n!/(n!*(n-k)!))
            Space: O(n!/(n!*(n-k)!))   (extra space: O(k))
            """
            comb, cur = [], []
            start = 1
            while True:
                l = len(cur)

                if l == k:
                    comb.append(cur[:])
                """
                k - l > n - start + 1 means that l will not satisfy k in the future
                in fact, (k - l) > (n - start + 1)  can cover start > n when (l-k) = -1
                The backtrack condition is the same as recursive approach1:
                1. len(cur) == k
                2. can not satisfity in the future: (k - l) > (n - start + 1)
                3. out of boundary
                """
                if l == k or (k - l) > (n - start + 1) or start > n:
                    if not cur: # done !
                        break
                    start = cur.pop() + 1
                else:
                    cur.append(start)
                    start += 1
            return comb
          ```
    * 039: Combination Sum (M)
      * Given a set of candidate numbers (candidates) (**without duplicates**) and a target number (target), **find all unique combinations** in candidates where the candidate numbers sums to target
      * Time:
        * https://leetcode.com/problems/combination-sum/discuss/16634/If-asked-to-discuss-the-time-complexity-of-your-solution-what-would-you-say
      * Approach1: Recursive:
        * Python
          ```python
          def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
            def dfs(start, target, cur):
                if target < 0:
                    return

                if target == 0:
                    results.append(cur[:])
                    return

                for i in range(start, n):
                    num = candidates[i]
                    cur.append(num)

                    """
                    use start the keep the sequence
                    """
                    dfs(i, target-num, cur)
                    cur.pop()

            if target < 0:
                return []

            n = len(candidates)
            results = []
            cur = []
            dfs(0, target, cur)
            return results
          ```
      * Approach2: Iterative + DP
        * Ref:
          * https://leetcode.com/problems/combination-sum/discuss/16509/Iterative-Java-DP-solution
    * 040: Combination Sum II (M)
      * Given a collection of candidate numbers (candidates) and a target number (target), **find all unique combinations** in candidates where the candidate numbers sums to target.
      * Each number in candidates **may only be used once** in the combination.
      * Approach1: BackTracking:
        * Python
          ```python
          def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
            def dfs(start, target, cur):
                if target < 0:
                    return

                if target == 0:
                    results.append(cur[:])
                    return

                for i in range(start, n):

                    # The val in the same position can be used only once.
                    if i > start and candidates[i] == candidates[i-1]:
                        continue
                    num = candidates[i]
                    cur.append(num)
                    dfs(i+1, target-num, cur)
                    cur.pop()


            if target < 0:
                return []

            candidates.sort()
            n = len(candidates)
            results = []
            cur = []
            dfs(0, target, cur)
            return results
          ```
    * 216: Combination Sum III (M)
    * 377: Combination Sum IV (M)
  * **Permutation**
    * 046: Permutations (M)
      * Given a collection of **distinct integers**, return all possible permutations.
      * Approach1: Recursive, Time: O(n!), Space: O(n!),
        * Time: O(n!)
          * Total: n * n-1 * n-2..  * 2 * 1  -> n!
        * Python
          ```python
          def permute(self, nums: List[int]) -> List[List[int]]:
            def backtrack(start):
                if start == n:
                    perms.append(nums[:])
                    return

                for i in range(start, n):
                    nums[start], nums[i] = nums[i], nums[start]
                    backtrack(start+1)
                    nums[start], nums[i] = nums[i], nums[start]

            if not nums:
                return []

            perms = []
            n = len(nums)
            backtrack(0)

            return perms
          ```
      * Approach2: Iterative, Time: O(n!), Space: O(n!)
        * Time: O(n!)
          * Total: 1 * 2 * 3 * ... (n-1) * n operation -> n!
        * Space: O(n!)
          * n! permutations
        * example: [a, b, c]
          * s1: [a]
          * s2: [a, b], [b, a]
          * s3: [c, a, b], [a, c, b], [a, b, c], [c, b, a], [b, c, a], [b, a, c]
        * Python
            ```python
            def permute(self, nums: List[int]) -> List[List[int]]:
              if not nums:
                  return []

              perms = [[nums[0]]]

              for i in range(1, len(nums)):
                  num = nums[i]
                  new_perms = []

                  for perm in perms:
                      for b in range(0, len(perm)+1):
                          new_perms.append(perm[:b] + [num] + perm[b:])

                  perms = new_perms

              return perms
            ```
    * 047: Permutations II (M)
      * Given a collection of numbers **that might contain duplicates**, return all possible unique permutations.
      * Note:
        * How to avoid duplicate val in the same position?
        * Length of Permutation should be n.
      * Approach1: Recursive, BackTracking + Counter
        * Note:
          * How to avoid duplicate val in the same position?
          * Length of Permutation should be n.
          * answer:
            * Use Counter to Control recursive call.
        * Python
          ```python
          def permuteUnique(self, nums):
              def backtrack(counter, cur):
                  if len(cur) == n:
                      results.append(cur[:])
                      return

                  # don't pick duplicates in the same position
                  for num in counter:
                      # take only once in each permuation
                      if counter[num] == 0:
                          continue

                      cur.append(num)
                      counter[num] -= 1

                      # pop the key when the number == 0 and append later ??
                      backtrack(counter, cur)

                      cur.pop()
                      counter[num] += 1

              n = len(nums)
              results = []
              cur = []
              counter = Counter(nums)
              backtrack(counter, cur)

              return results
          ```
      * Approach2: Iterative bottom up
        * Ref
          * https://leetcode.com/problems/permutations-ii/discuss/18602/9-line-python-solution-with-1-line-to-handle-duplication-beat-99-of-others-%3A-)
        * How to avoid duplicate ?
          * just avoid inserting a number AFTER any of its duplicates.
          * Another way to think is that we have symmetry and inserting only towards left of previous 2 or only towards right of previous 2 is all we have to do.
            * 1. [n]
            * 2. [n* n] == [n n*]
          * Example:
            ```txt
            [1, 2, 2*]

            step1:
            [1]

            step2:
            [1, 2],  [2, 1]

            step3:
            [2*, 1, 2],  [1, 2*, 2],  x[1, 2, 2*]
            [2*, 2, 1], x[2, 2*, 1],  x[2, 1, 2*]
            ``
        * Python
          ```python
          def permuteUnique(self, nums: List[int]) -> List[List[int]]:
            if not nums:
                return []

            perms = [[]]
            for n in nums:
                new_perms = []
                for perm in perms:
                    for i in range(len(perm)+1):
                        new_perms.append(perm[:i]+[n]+perm[i:])
                        # To handle duplication
                        # just avoid inserting a number AFTER any of its duplicates.
                        if i < len(perm) and perm[i] == n:
                            break

                perms = new_perms
            return perms
          ```
    * 031: Next Permutation (M)
    * 060: Permutation Sequence (M)
  * 291: Word Pattern II
### Graph
  * Eulerian trail
    * A finite graph that **visits every edge** exactly once.
    * 332: Reconstruct Itinerary (M)
      * This is the problem of Eulerian trail
        * vertex: airport
        * edge: ticket
      * FAQ
        * how do you make sure there is no dead end since you always choose the "smallest" arrivals (min heap) ?
          * Starting at the first node, **we can only get stuck at the ending point, since every node except for the first and the last node has even number of edges, when we enter a node we can always get out**.
          * Now we are at the destination
            * case1: if all edges are visited, we are done, and the dfs returns to the very first state.
            * case2: Otherwise** we need to "insert" the unvisited loop into corresponding position, and in the dfs method, it returns to the node with extra edges, **starts another recursion and adds the result before the next path**. This process continues until all edges are visited.
          * Example:
            * A Graph:, start is A, end is E
              * A -> [B, E]  # start vertex
              * B -> [C]
              * C -> [A]
              * E -> []      # end vertex
            * Case 1:
              * s1: A
              * s2: A -> **B->C->A->E** (Iterative B end)
            * Case 2:
              * s1: A
              * s2: A -> **E** (iterative E end)
              * s2: A -> **B->C->A** (Iterative B end) -> E
      * Approach1: Hierholzer's algorithm, DFS Recursive, Time:O(Elog(E)), Space:O(E)
        * Time: O(ELog(E))
          * Generate Adjacency List: O(ELog(E))
          * Traverse every edge: O(ELog(E))
        * Space: O(E)
          * Adjacency List: O(E)
          * Recursive Call: O(E)
        * Python Solution:
          ```python
          def findItinerary(self, tickets: List[List[str]]) -> List[str]:
            def _dfs(departure):
                arrival_heap = flights[departure]
                while arrival_heap:
                    _dfs(heapq.heappop(arrival_heap))

                # apeendleft to avoid unnecessary reverse
                route.appendleft(departure)

            flights = collections.defaultdict(list)
            route = collections.deque()

            for departure, arrivial in tickets:
                arrival_heap = flights[departure]
                # put the neighbors in a min-heap for lexical order
                heapq.heappush(arrival_heap, arrivial)

            # start from JFK
            _dfs("JFK")

          return route
          ```
      * Approach2: Hierholzer's algorithm, DFS Iterative Time:O(Elog(E)), Space:O(E)
        * Time: O(ELog(E))
          * Generate Adjacency List: O(ELog(E))
          * Traverse every edge: O(ELog(E))
        * Space: O(E)
          * Adjacency List: O(E)
          * Stack: O(E)
        * Python Solution
          ```python
          def findItinerary(self, tickets: List[List[str]]) -> List[str]:
            flights = collections.defaultdict(list)

            for departure, arrivial in tickets:
                arrival_heap = flights[departure]
                # put the neighbors in a min-heap for lexical order
                heapq.heappush(arrival_heap, arrivial)

            stack = ["JFK"]  # start from JFK
            route = collections.deque()

            while stack:
                departure = stack[-1]
                arrival_heap = flights[departure]
                if arrival_heap:  # do not use while here
                    stack.append(heapq.heappop(arrival_heap))
                else:
                    # apeendleft to avoid unnecessary reverse
                    route.appendleft(stack.pop())

            return route
          ```
  * Eulerian circuit
    * An **Eulerian trail** that **starts and ends on the same vertex**.
  * Hamilton path
    * A finite graph that **visits every vertex** exactly once.
  * Hamilton cycle
    * An Hamilton path that **starts and ends on the same vertex**.
  * Minimum Spanning Tree
    * A minimum spanning tree (MST) or minimum weight spanning tree is **a subset of the edges of a connected, edge-weighted undirected graph that connects all the vertices together, without any cycles and with the minimum possible total edge weight**.
  * Shortest Path
  * Topological Sort
    * Ref:
      * https://www.youtube.com/watch?v=ddTC4Zovtbc
      * https://leetcode.com/problems/course-schedule-ii/solution/
    * 207: Course Schedule (M)
      * Approach1: Peeling Onion, Time: O(V+E), Space: O(V+E)
        * Time: O(V+E)
          * Build Outdegree List Graph and Indegree Array
            * O(E)
          * Traverse the Nodes and Edges: O(V+E)
            * Traverse the Nodes
                * Each node would be traverse at most once.
                * Operation for push node to queue and pop node from queue.
            * Traverse the edges.
              * Each edge would be traversed at most once.
              * Operation for **removing outdegree of the node**.
        * Space:O(V+E)
          * Graph of adjency list
            * O(V+E)
          * Indegree Array
            * O(V)
          * BFS Queue
            * O(V)
        * Algorithm
          * Definition:
            * **Indegree of the Course:**
              * How many prequisites courses of it.
            * **Outdegree of Course**
              * How many courses' prequisite is this course.
          * The algorithm
            * We first process all the nodes/course with **0 in-degree** implying no prerequisite courses required. **If we remove all these courses from the graph, along with their outgoing edges**, we can find out the courses/nodes that should be processed next. These would again be the nodes with 0 in-degree. We can continuously do this until all the courses have been accounted for.
          * Data Structures
            * **A graph of adjency list with indegree**
              * If course u is a prerequisite of course v, then the adjacency list of u will contain v.
            * **A degree array**
              * Calculate how many prerequisite courses for each.
        * Python Solution:
          ```python
          def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
              """
              graph of adjency list:
              If course u is a prerequisite of course v,
              then the adjacency list of u will contain v.
              """
              # outdegree Adjacency list, the list stores nodes of outdegree
              g_adj = [[] for _ in range(numCourses)]

              # indegree array
              a_indegree = [0 for _ in range(numCourses)]

              for course, preq_course in prerequisites:
                  g_adj[preq_course].append(course)
                  a_indegree[course] += 1

              q = collections.deque()
              for course, indegree in enumerate(a_indegree):
                  # indegree == 0 means no prequisites
                  if indegree == 0:
                      q.append(course)

              sheduled_cnt = 0
              while q:
                  cur = q.popleft()
                  sheduled_cnt += 1
                  # remove outdegree edges of node
                  for next_course in g_adj[cur]:
                      a_indegree[next_course] -= 1
                      # indegree == 0 means no prequisites
                      if a_indegree[next_course] == 0:
                          q.append(next_course)

              return sheduled_cnt == numCourses
          ```
      * Approach2: DFS:
        * Algorithm:
          * For each of the nodes in our graph, we will run a depth first search in case that node was not already visited in some other node's DFS traversal.
          * Suppose we are executing the depth first search for a node N. We will recursively traverse all of the neighbors of node N which have not been processed before.
          * Once the processing of all the neighbors is done, we will add the node N to the stack. We are making use of a stack to simulate the ordering we need. When we add the node N to the stack, all the nodes that require the node N as a prerequisites (among others) will already be in the stack.
        * Python Solution
          ```python
          class Status(object):
            WHITE = 1  # default
            GRAY =  2  # processing
            BLACK = 3  # done

          class Solution:
              def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
                  is_cyclic = False
                  scheduled_course = []

                  def dfs(course):
                      # !!! don't forget use nonlocal
                      nonlocal is_cyclic

                      if is_cyclic:
                          return

                      visits[course] = Status.GRAY

                      for next_course in g_adj[course]:
                          if visits[next_course] == Status.WHITE:
                              dfs(next_course)

                          # graph cucle found
                          elif visits[next_course] == Status.GRAY:
                              is_cyclic = True
                              return
                      # done
                      visits[course] = Status.BLACK
                      scheduled_course.append(course)

                  # Adjacency list of outdegree
                  g_adj = [[] for _ in range(numCourses)]
                  visits = [Status.WHITE for _ in range(numCourses)]

                  for course, preq in prerequisites:
                      g_adj[preq].append(course)

                  # perhaps start from indegree 0 ??
                  for course in range(numCourses):
                      if visits[course] == Status.WHITE:
                          dfs(course)

                  return scheduled_course[::-1] if not is_cyclic else []
          ```
    * 210: Course Schedule II (M)
      * Same Concept as 207, the only difference is to keep the scheduled courses list.
      * Python Solution
        ```python
        def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

            # graph Adjacency list, the list stores nodes of outdegree
            g_adj = [[] for _ in range(numCourses)]

            # indegree array
            a_indegree = [0 for _ in range(numCourses)]

            for course, preq_course in prerequisites:
                g_adj[preq_course].append(course)
                a_indegree[course] += 1

            q = collections.deque()
            for course, indegree in enumerate(a_indegree):
                # indegree == 0 means no prequisites
                if indegree == 0:
                    q.append(course)

            scheduled_course = list()
            while q:
                cur = q.popleft()
                scheduled_course.append(cur)

                for nxt in g_adj[cur]:
                    a_indegree[nxt] -= 1
                    if a_indegree[nxt] == 0:
                        q.append(nxt)

            if len(scheduled_course) == numCourses:
                return scheduled_course
            else:
                return []
        ```
    * 269: Alien Dictionary (H)
### Bit Manipulation
### Union Field
  * 1135: Connecting Cities With Minimum Cost
  * 200: Number of Islands (M)
  * 305: Number of Islands II (H)
### Retry List
  * Math:
    * Number:
      * 007: Reverse Integer
        * Boundary and negative value
      * 202: Happy Number (E)
      * Excel Sheet
        * 168: **Excel Sheet** Column Title
    * Majority
      * Boyer-Moore Voting Algorithm:
        * notice the concpet and implementation.
      * 1150: Check If a Number Is Majority Element in a Sorted Array (E)
      * 229: Majority Element II (M)
    * Pascal's Triangle
      * 119: Pascal's Triangle II (E)
    * Sum
      * 015: 3Sum (M)
      * 018: 4Sum (M) and kSum
  * Sorting
    * quick sort recursiv + iterative
    * merge sort recursiv + iterative:
      * using indices in the merge function.
    * heap sort
      * Use max heap
    * 075: Sort Colors (M)
      * Notice the end conditino of Dutch National Flag Problem
  * String
    * 038: Count and Say (E)
    * 392: Is Subsequence (E)
    * substring
      * 028: Implement **strStr** (E) (**KMP** algorithm)
      * 003: Longest Substring Without Repeating Characters (M)
      * 076: Minimum Window Substring (H)
    * Anagram
      * 049: Group Anagrams (M)
      * 249: Group Shifted Strings (M)
    * Number and carry:
      * 168: Excel Sheet Column Title (E)
      * 013: Romain to Integer (E)
    * Other:
      * 014: Longest Common Prefix (E)
      * 294: Flip Game II (M)
  * Array
    * Container
      * 042: Trapping Rain Water (H)
        * Don't forget why we should include height[i] for left_max and right_max array
          * example:
            * [2, 10, 5], for 1th bin, if we do not inlcude 10, the area would be negative.
    * Jump Game
      * 045: Jump Game II (H)
    * 325: Maximum Size Subarray Sum **Equals k** (M)
    * 560: Subarray Sum **Equals K** (M)
    * 152: Maximum **Product** Subarray (M)
    * 134: Gas Station (M)
  * LinkedList
    * **Runner** and **Detect Circle**
      * 142: Linked List Cycle II (M)
    * Remove node:
      * 082: Remove Duplicates from **Sorted** List II (M)
        * don't forget the last delete_from_prev
    * **Reorder**:
      * 092: **Reverse** Linked List II (M)
      * 025: **Reverse** Nodes in **k-Group** (H)
      * 024: **Swap** Nodes in **Pair** (M)
      * 143: **Reorder** List (M)
    * Partition:
      * 061: **Rotate** list (M)
      * 328: Odd Even Linked List (M)
      * 725: Split Linked List in Parts (M)
    * **Sorting and Merge**
      * 023: Merge k **Sorted** Lists (H)
      * 148: Sort list (M)
      * 147: Insertion Sort List (M)
    * Add numbers
      * 002: Add Two Numbers (M)
      * 369: Plus One Linked List (M)
  * Tree:
    * PreOrder:
      * 111: **Minimum Depth** of Binary Tree	(E)
      * 298: Binary Tree **Longest Consecutive** Sequence (M)
    * InOrder:
      * 105: Construct Binary Tree from **Preorder** and **Inorder** Traversal (M)
      * 106: Construct Binary Tree from **Inorder** and **Postorder** Traversal	(M)
    * PostOrder:
      * 145: Binary Tree **Postorder** Traversal (H)
      * 110: **Balanced** Binary Tree (E)
      * 124: Binary Tree **Maximum Path Sum** (H)
      * 250: Count **Univalue Subtrees** (M)
      * 366: **Find Leaves** of Binary Tree (M)
      * 236: **Lowest Common Ancestor** of a Binary Tree (M)
    * BFS & DFS:
      * 107: Binary Tree Level Order Traversal II
        * Recursive
      * 113: Path **Sum** II (M)
      * 314: Binary Tree **Vertical Order** Traversal	(M)
      * 101: **Symmetric** Tree (E)
      * 297: **Serialize** and **Deserialize** Binary Tree (H)
      * 116 & 117: **Populating Next Right Pointers** in Each Node (M)
    * Binary Search Tree (BST)
      * 098: Validate Binary Search Tree (M)
      * 450: **Delete** Node in a BST (M)
      * 235: **Lowest Common Ancestor** of a Binary Search Tree (E)
      * 108: **Convert Sorted Array** to Binary Search Tree	binary search (E)
      * 285: **Inorder Successor** in BST (M)
      * 510: **Inorder Successor** in BST II (M)
      * 270: **Closest** Binary Search Tree Value (E)
      * 272: **Closest** Binary Search Tree Value II (H)
      * 096: **Unique** Binary Search **Trees**	(M)
        * DP
      * 255: Verify Preorder Sequence in Binary Search Tree (M)
  * Binary Search:
    * 035: Search Insert Position (E)
    * 278: First Bad Version (E)
    * 275: H-Index II (M)
    * 033 & 081: Search in Rotated Sorted Array (M)
    * 004: Median of Two Sorted Arrays (H)
    * 222: Count **Complete** Tree Nodes (M)
    * 162: Find Peak Element (M)
  * Cache
    * 146: LRU Cache (M)
  * Heap
    * 692: Top K Frequent Words (M)
  * BackTracking
    * 022: Generate Parentheses (M)
    * 077: Combinations (M)
    * 078: Subsets (M)
  * BFS & DFS
    * 200: Number of Islands (M)
      * For BFS / DFS iterations, set visits to True before append to the queue to **reduce unnecessary queue/stack push/pop operations.**
    * 286: Walls and Gates (M)
      * Start From Gates
    * 364: **Nested List** Weight Sum II (M)
    * 127: Word Ladder (M)
  * Graph
    * Eulerian trail
      * 332: Reconstruct Itinerary (M)
    * Topological Sort
      * 207: Course Schedule (M)
  * Dynamic Programming:
    * Fibonacci sequence
      * 070: Climbing Stairs
      * 091: **Decode Ways**
    * **One Dimensional**
    * **Two Dimensional**
      * 256: Paint House (E)
      * 072: Edit Distance (H)
    * **Deduction**
      * 276: Paint Fence (E)
      * 198: House Robber (E)
      * 213: House Robber II (M)
      * 337: House Robber III (M)


