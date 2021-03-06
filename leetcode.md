Table of Content
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
  - [Union Find (Disjoint-set)](#union-find-disjoint-set)
  - [Backtracking](#backtracking)
  - [Graph](#graph)
  - [Bit Manipulation](#bit-manipulation)
  - [Dynamic Programming](#dynamic-programming)
  - [Retry List](#retry-list)



## [Classification](https://cspiration.com/leetcodeClassification#103)
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
              boundary = (2 ** 31)//10

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
                  # once
                  slow = digit_square_sum(slow)
                  # twice
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

              while n :
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
                    cur_max = score
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
        * Approach1, Time:O(n), Space:O(c)
          * Python
            ```python
            def isStrobogrammatic(self, num):
              d = {'6':'9', '9':'6', '8':'8', '1':'1', '0':'0'}

              is_strobogrammatic = True
              l, r = 0, len(num)-1
              while l <= r:
                  # if left == right:
                  # the valid cases would be '8':'8', '1':'1', '0':'0'
                  n_l, n_r = num[l], num[r]
                  if n_l not in d or n_r not in d or d[n_l] != n_r:
                    is_strobogrammatic = False
                    break

                  l += 1
                  r -= 1
              return is_strobogrammatic
            ```
      * 247 Strobogrammatic Number II (M)
        * Description:
          * Find all strobogrammatic numbers that are of length = n.
        * Approach1, backtracking, Time: O(nk^n), Space:O(nk^n)
          * Python
            ```python
            def findStrobogrammatic(self, n: int) -> List[str]:

              def backtrack(cur, idx):
                  if idx == n // 2:
                      combs.append("".join(cur))
                      return

                  append_keys = D_NON_FINAL
                  if idx == n//2-1:
                      append_keys = D_FINAL

                  for l, r in append_keys.items():
                      cur.append(r)
                      cur.appendleft(l)
                      backtrack(cur, idx+1)
                      cur.pop()
                      cur.popleft()

              if n < 1:
                  return []

              D_NON_FINAL = {'6':'9', '9':'6', '8':'8', '1':'1', '0':'0'}
              D_FINAL = {'6':'9', '9':'6', '8':'8', '1':'1'}
              combs = []
              cur = collections.deque()
              if n % 2:
                  mids= ["0", "1", "8"]
                  for m in mids:
                      cur.append(m)
                      backtrack(cur, 0)
                      cur.pop()
              else:
                  backtrack(cur, 0)

              return combs
            ```
        * Approach2, Bottom up iterative: Time: O(nk^n), Space:O(nk^n)
          * Time: O(nk^n)
            * Total n//2 iterations:
              * for each iteration:
                * k * prev_combo cnt operations
            * At first, 1 or 3 combinations
            * Finally, about k^n combinations
            * each combination cost O(n) to copy
            * (n * k^n) // 2  = O(nk^n)
          * Space: O(nk^n)
            * About O(k^n) combinations
            * each combination needs O(n) space
          * Python:
            ```python
            def findStrobogrammatic(self, n: int) -> List[str]:
              if n < 1:
                  return []

              D_NON_FINAL = {'6':'9', '9':'6', '8':'8', '1':'1', '0':'0'}
              D_FINAL = {'6':'9', '9':'6', '8':'8', '1':'1'}

              if n % 2:
                  combs = ["0", "1", "8"]
              else:
                  combs = [""]

              for i in range(n//2):
                  append_keys = D_NON_FINAL
                  if i == n//2-1:
                      append_keys = D_FINAL

                  next_combs = []
                  for comb in combs:
                      for l, r in append_keys.items():
                          next_combs.append(f"{l}{comb}{r}")

                  combs = next_combs

              return combs
            ```
      * 248 Strobogrammatic Number II (H)
        * Write a function to count the total strobogrammatic numbers that exist in the range of low <= num <= high.
    * 273: Integer to **English Words** (H)
    * 065: Valid Number (H)
  * **kSum**
    * 001: Two Sum (E)
      * Approach1: hash table, Time: O(n), Space: O(n)
        * Python
          ```python
          def twoSum(self, nums: List[int], target: int) -> List[int]:
            """
            Assume exactly one solution
            """
            res = None
            d = dict()
            for idx, num in enumerate(nums):
                diff = target - num
                if diff in d:
                    res = [d[diff], idx]
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
        * Python
            ```python
            def threeSum(self, nums: List[int]) -> List[List[int]]:
              def two_sum(l, r, target, cur):
                  while l < r:
                      s = nums[l] + nums[r]
                      if s == target:
                          combs.append(cur + [nums[l], nums[r]])
                          while l < r and nums[l] == nums[l+1]:
                              l += 1
                          while l < r and nums[r] == nums[r-1]:
                              r -= 1
                          l += 1
                          r -= 1
                      elif s < target:
                          l += 1
                      else:
                          r -= 1

              n = len(nums)
              nums.sort()
              combs = []
              cur = []
              for i in range(n-2):
                  if i > 0 and nums[i] == nums[i-1]:
                      continue
                  cur.append(nums[i])
                  two_sum(l=i+1,
                          r=n-1,
                          target=0-nums[i],
                          cur=cur)
                  cur.pop()
              return combs
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
                    # boundary: r+1 (exclusive)
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
      * Approach3: ksum iterative ?
    * 016: 3Sum Closest (M)
      * Approach1: O(n^2), Space:O(1)
        * Python
          ```python
          def threeSumClosest(self, nums: List[int], target: int) -> int:
            def two_sum(l, r, target, cur):
                nonlocal min_diff
                nonlocal comb_sum
                while l < r:
                    s = nums[l] + nums[r]
                    diff = abs(target-s)
                    if diff < min_diff:
                        min_diff = diff
                        comb_sum = sum(cur) + nums[l] + nums[r]

                    if s == target:
                        break
                    elif s < target:
                        l += 1
                    else:
                        r -= 1

            n = len(nums)
            nums.sort()
            cur = []
            min_diff, comb_sum = float('inf'), None
            for i in range(n-2):
                if i > 0 and nums[i] == nums[i-1]:
                    continue
                cur.append(nums[i])
                two_sum(l=i+1,
                        r=n-1,
                        target=target-nums[i],
                        cur=cur)
                cur.pop()

                if min_diff == 0:
                    break

            return comb_sum
          ```
  * **Majority**
    * Definition:
      * The majority element is **the element that appears more than ⌊ n/2 ⌋ times**.
    * 1150: Check If a Number Is Majority Element in a Sorted Array (E)
      * See Binary Search
    * 169: Majority Element (E)
      * Description:
        * You may assume that the array is non-empty and the majority element always exist in the array.
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

            pascal = [[1]]
            for layer in range(2, numRows+1):
                new = [1] * layer
                prev = pascal[-1]
                for i in range(1, layer-1):
                    new[i] = prev[i-1] + prev[i]
                pascal.append(new)

            return pascal
          ```
    * 119: Pascal's Triangle II (E)
      * Get kth Pascal index
      * Approach1: From end to beginning: Time:O(n^2), Space:O(k)
        * Python
          ```python
          def getRow(self, rowIndex: int) -> List[int]:
            num_row = rowIndex + 1
            if num_row == 0:
                return []

            pascal = [1] * num_row

            if num_row <= 2:
                return pascal

            for layer in range(3, num_row+1):
                # from layer-2 to 1
                for i in range(layer-2, 0, -1):
                    pascal[i] = pascal[i] + pascal[i-1]

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
                if len(nums) < 2:
                  return nums

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

              if len(nums) < 2:
                return nums

              insertion_sort(len(nums)-1)
              return nums
            ```
      * Approach2: **Selection** Sort, Time:O(n^2), Space:O(1)
        * Select min from 0 to n-2
        * Iterative: Time:O(n^2), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              if len(nums) < 2:
                  return nums

              n = len(nums)
              # 0 ~ n-2
              for select in range(0, n-1):
                  # select + 1 ~ n
                  mini = select
                  for i in range(select+1, n):
                      if nums[i] < nums[mini]:
                          mini = i
                  nums[mini], nums[select] = nums[select], nums[mini]

              return nums
            ```
        * Recursive: Time:O(n^2), Space:O(n)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              def selection_sort(select):
                  if select > n-2:
                      return

                  mini = select
                  for i in range(select+1, n):
                      if nums[i] < nums[mini]:
                          mini = i
                  nums[mini], nums[select] = nums[select], nums[mini]

                  selection_sort(select + 1)

              if len(nums) < 2:
                  return nums

              n = len(nums)
              selection_sort(0)
              return nums
            ```
      * Approach3: **Bubble** Sort, Time:O(n^2), Space:O(1)
        * Bubble max to the end from n-1 to 1
        * Iterative: Time:O(n^2), Space:O(1)
          * Python
            ```python
            def sortArray(self, nums: List[int]) -> List[int]:
              if len(nums) < 2:
                  return nums

              n = len(nums)

              # n-1 to 1
              for bubble in range(n-1, 0, -1):
                  # 0 to bubble-1
                  is_swap = False
                  for i in range(0, bubble):
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
                  for i in range(0, bubble):
                      if nums[i] > nums[i+1]:
                          nums[i], nums[i+1] = nums[i+1], nums[i]
                          is_swap = True

                  if not is_swap:
                      return

                  bubble_sort(bubble-1)


              if len(nums) < 2:
                  return nums

              n = len(nums)
              bubble_sort(n-1)
              return nums
            ```
      * Approach4: **Quick** Sort, Time:O(nlogn), Space:O(logn~n)
        * complexity:
          * In the quicksort, you have to take care of two sides of the pivot. But in quickselect, you only focus on the side the target object should be. So, in optimal case, the running time of quicksort is (n + 2 * (n/2) + 4 * (n/4)...), it has logn iterations. Instead, the running time of quickselect would be (n + n/2 + n/4 +...) it has logn iterations as well
        * quick select:
          * Python
            ```python
            def partition(nums, start, end):
              ran = random.randint(start, end)
              pivot = end
              nums[pivot], nums[ran] = nums[ran], nums[pivot]

              border = start
              for cur in range(start, end):
                  if nums[cur] <= nums[pivot]:
                    nums[cur], nums[border] = nums[border], nums[cur]
                    border += 1

              nums[border], nums[pivot] = nums[pivot], nums[border]
              return border

            def quick_select(nums, start, end, k_smallest):
                res = None
                while start <= end:
                    p = partition(nums, start, end)
                    if p == k_smallest:
                        res = nums[k_smallest]
                        break
                    elif p > k_smallest:
                        end = p - 1
                    else:
                        start = p + 1
                return res
            ```
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

              n = len(nums)
              stack = [(0, n-1)]
              while stack:
                  left, right = stack.pop()
                  p = get_partition(left, right)

                  if left < p-1:
                      stack.append((left, p-1))

                  if p+1 < right:
                      stack.append((p+1, right))

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

              def quick_sort(left, right):
                  p = get_partition(left, right)

                  if left < p-1:
                      quick_sort(left, pivot-1)

                  if p+1 < right:
                      quick_sort(pivot+1, right)

              if len(nums) < 2:
                  return nums

              n = len(nums)
              quick_sort(0, n-1)
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

                  left = nums[left_start: left_end+1]
                  right = nums[right_start: right_end+1]
                  d = dst_start
                  l = r = 0
                  while l < len(left) and r < len(right):
                      if left[l] < right[r]:
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

              if len(nums) < 2:
                  return nums

              n = len(nums)
              win_size = 1
              while win_size < n:
                  left = 0
                  while left + win_size < n:
                      mid = left + win_size
                      right = mid + win_size
                      merge_2_sorted_array(left,
                                          left, mid-1,
                                          mid, right-1)

                      left = right
                  win_size *= 2

              return nums
            ```
        * Recursive
          ```python
          def sortArray(self, nums: List[int]) -> List[int]:
            def merge_2_sorted_array(dst_start,
                                    left_start, left_end,
                                    right_start, right_end):

                left = nums[left_start: left_end+1]
                right = nums[right_start: right_end+1]
                d = dst_start
                l = r = 0
                while l < len(left) and r < len(right):
                    if left[l] < right[r]:
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
                if not left < right:
                    return

                mid = (left + right) // 2
                merge_sort(left, mid)
                merge_sort(mid+1, right)
                merge_2_sorted_array(left,
                                     left, mid,
                                     mid+1, right)

            if len(nums) < 2:
                return nums

            n = len(nums)
            merge_sort(0, n-1)

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
                      """
                      left child and right child
                      """
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
    * bucket sort:
      * see 347: **Top K Frequent** Elements (M)
### String
  * Remove Duplicate:
    * 316: Remove Duplicate Letters (H)
  * Text
    * 068: Text Justification (H)
  * **Edit Distance**:
    * 161: One Edit Distance (M)
       * Notice the **zero edit distance cases**.
       * Approach1: Time O(n+m), Space (1):
         * Merge the insert and remove cases (find the short one)
         * Use short and long string pointers to traverse and compare
         * Python
          ```python
          def isOneEditDistance(self, s: str, t: str) -> bool:
            diff = len(s) - len(t)
            if abs(diff) >= 2:
                return False

            same_len = False
            if diff < 0:
                short, long = s, t
            elif diff == 0:
                short, long = s, t
                same_len = True
            else:
                short, long = t, s

            is_one_edit = True
            edit_cnt = 0
            i = j = 0
            while i < len(short):
                if short[i] == long[j]:
                    i, j = i+1, j+1
                    continue

                if edit_cnt != 0:
                    is_one_edit = False
                    break

                edit_cnt += 1
                if same_len:  # edit operation
                    i, j = i+1, j+1
                else:         # insert operation
                    j += 1

            if same_len and edit_cnt == 0:
                is_one_edit = False

            return is_one_edit
          ```
    * 072: Edit Distance (H)
      * See DP
  * SubString:
     * 459: Repeated Substring Pattern
       * Description:
         * Given a non-empty string check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.
       * Approach1: Brute Force, Time:(n^2), Space:O(1)
       * Approach2: Time:O(n), Space:O(n)
         * Ref:
           * https://leetcode.com/problems/repeated-substring-pattern/discuss/94334/Easy-python-solution-with-explaination
         * If the string S has repeated block, it could be described in terms of pattern.
           * S = SpSp (For example, S has two repeatable block at most)
           * If we repeat the string, then SS=SpSpSpSp.
           * Destroying first and the last pattern by removing each character, we generate a new S2=SxSpSpSy.
           * If the string has repeatable pattern inside, S2 should have valid S in its string.
         * Python
          ```python
          def repeatedSubstringPattern(self, s: str) -> bool:
            return s in (2 * s)[1:-1]
          ```
     * 030: Substring with Concatenation of All Words (H)
     * 395: Longest Substring with **At Least K Repeating** Characters (M)
     * 340: Longest Substring with At Most K Distinct Characters (H)
     * 159: Longest Substring with At Most Two Distinct Characters (H)
  * **Sliding Window:**
     * 643: Maximum Average Subarray I (E)
        * Description:
          * Given an array consisting of n integers, find the **contiguous subarray of given length k** that has the maximum average value.
        * Approach1: Sliding window, Time:O(n), Space:O(1)
          * Python
            ```python
            def findMaxAverage(self, nums: List[int], k: int) -> float:
              if not nums:
                  return float(0)

              if k == 1:
                  return float(max(nums))

              max_avg = float('-inf')
              cur_sum = float(0)
              for i, n in enumerate(nums):
                  cur_sum += n

                  # maintain the window with size k
                  if (i + 1) >= k:
                      max_avg = max(max_avg, cur_sum/k)
                      cur_sum -= nums[i-k+1]

              return max_avg
            ```
     * 003:	Longest Substring Without Repeating Characters (M)
       * Description:
         * Given a string, find the length of the longest substring without repeating characters.
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
         * Python:
            ```python
            def lengthOfLongestSubstring(self, s: str) -> int:
              start = end = 0
              d = dict()
              max_len = 0

              while end < len(s):
                  c = s[end]

                  if c in d and d[c] >= start:
                      # move the left boundary
                      start = d[c] + 1

                  max_len = max(max_len, end-start+1)
                  d[c] = end
                  end += 1

              return max_len
            ```
     * 076: Minimum Window Substring (H)
       * Description:
         * Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
       * Approach1: Brute Force, Time:(n^2), Space:O(1)
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
              txt, pattern = s, t

              if not txt or not pattern:
                  return ""

              """
              formed is used to keep track of how many unique characters in t are present in the
              current window in its desired frequency.
              e.g. if t is "AABC" then the window must have two A's, one B and one C.
                    Thus formed would be = 3 when all these conditions are met.
              """
              COUNTER_P = collections.Counter(pattern)
              FORMED_P = len(COUNTER_P)

              counter_t = collections.defaultdict(int)
              formed_t = 0

              UPPER_BOUND = len(s) + 1
              res = [UPPER_BOUND, None, None]

              start = 0
              for end in range(len(txt)):
                  c = txt[end]
                  if c in COUNTER_P:
                      counter_t[c] += 1
                      if counter_t[c] == COUNTER_P[c]:
                          formed_t += 1

                  """
                  current window meets the requiremrnt
                  """
                  while start <= end and formed_t == FORMED_P:
                      window_len = end - start + 1
                      if window_len < res[0]:
                          res[0], res[1], res[2] = window_len, start, end

                      # try to shrink the window
                      r_c = txt[start]
                      if r_c in COUNTER_P:
                          counter_t[r_c] -= 1
                          if counter_t[r_c] < COUNTER_P[r_c]:
                              formed_t -= 1
                      start += 1

              return "" if res[0] == UPPER_BOUND else s[res[1]: res[2]+1]
            ```
     * 209: Minimum Size Subarray Sum (M)
        * Description:
          * Given an array of n positive integers and a positive integer s.
          * Find the **minimal length of a contiguous subarray of which the sum ≥ s**.
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

              UPPER_BOUND = len(nums) + 1
              min_len = UPPER_BOUND
              acc = 0
              start = 0
              for end in range(len(txt)):
                  acc += nums[end]

                  while start <= end and acc >= s:
                      w_len = end - start + 1
                      min_len = min(min_len, w_len)
                      # shrink the window
                      acc -= nums[start]
                      start += 1

              return 0 if min_len == UPPER_BOUND else min_len
            ```
     * 713: Subarray Product Less Than K (M)
       * Description:
        * Your are given an array of **positive integers nums**.
        * Count and print the number of (**contiguous**) subarrays where the product of all the elements in the subarray is less than k.
       * Approach1: Brute Force, Time:O(n^2), Space:O(1)
       * Approach2: Sliding window, Time:O(n), Space:O(1)
         * The challenge is how to add all subarray once.
         * **It would be better to think about the subarray is ended with right.**
         * Example:
          ```
          step1:
            [1, 2, 3, 4]
             lr
             l = r = 0
             all contiguous subarray ending with 1 and product < k is 1
             [1]
          step2:
            [ 1, 2, 3, 4]
              l  r
              l = 0, r = 1
             all contiguous subarray ending with 2 and < k is 2
             [2]
             [1,2]
          step3:
            [ 1, 2, 3, 4]
              l      r
              l = 0, r = 2
             all contiguous subarray ending with 3 and < k is 3
             [3]
             [2,3]
             [1,2,3]
          step4:
            [ 1, 2, 3, 4]
              l  l`    r
              l = 0, r = 3 but prod >= k
              move left until find new left whose produt < k
              if new left is l`
              all contiguous subarray ending with 4 and < k is 3
              [4]
              [3,4]
              [2,3,4]
          ```
         * Python
          ```python
          def numSubarrayProductLessThanK(self, nums, k):
            if k <= 1:
                return 0

            left = right = 0
            prod = 1
            cnt = 0
            while right < len(nums):
                prod *= nums[right]

                while left <= right and prod >= k:
                    prod /= nums[left]
                    left += 1

                # It would be better to think about the subarray is endied with right
                cnt += (right - left + 1)
                right += 1

            return cnt
          ```
     * 644: Maximum Average Subarray II (H)
       * Description:
         * Given an array consisting of n integers, find the contiguous subarray whose length is greater than or equal to k that has the maximum average value. And you need to output the maximum average value.
     * 239: Sliding Window Maximum (H)
  * **KMP**:
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
              * n at position 1
              * n, na at position 2
              * n, na, nap at position 3
              * n ,na nap, nape at position 4
        * Definition of the value in prefix suffix table
          * **The length of the longest proper prefix** in the (sub)pattern that matches a proper suffix in the same (sub)pattern.
      * Example
        * ![KMP example](./image/algo/KMP.png)
          * When un-macth happends, **the max reusable string range for next iteration is in suffix range**.
          * Seach from LPS array and find the proper start position of pattern comparison pointer (in this example, index 3).
    * 028: Implement **strStr** (E)
      * Approach1: Brute Force, Time: O(mn), Space(1)
         * Python
          ```python
          def strStr(self, haystack: str, needle: str) -> int:
              txt, pattern = haystack, needle
              res = NOT_FOUND = -1

              if not txt and not pattern:
                  return 0

              if not pattern:
                  return 0

              if len(pattern) > len(txt):
                  return NOT_FOUND

              for start in range(len(txt)-len(pattern)+1):
                  t, p = start, 0
                  for p in range(len(pattern)):
                      if txt[t] != pattern[p]:
                          break

                      t += 1
                      p += 1
                      if p == len(pattern):
                          res = t - len(pattern)
                          break

                  if res != NOT_FOUND:
                      break

              return res
            ```
      * Approach2: **KMP (substring match)**, Time: O(m+n), Space: O(n)
         * Python
           ```python
           class Solution:
            def get_lps(self, pattern):
                if len(pattern) == 1:
                    return [0]

                lps = [0] * len(pattern)
                p, s = 0, 1
                while s < len(lps):
                    if pattern[p] == pattern[s]:
                        p += 1
                        lps[s] = p
                        s += 1
                    else:
                        if p == 0:
                            """
                            do not match anything
                            lps[s] = 0
                            """
                            s += 1
                        else:
                          """
                          Reuse the prefix string that has been scanned.
                          The length of the longest common prefix suffix
                          are put in lps[p-1]
                          """
                          p = lps[p-1]

                return lps

              def strStr(self, haystack: str, needle: str) -> int:
                  txt, pattern = haystack, needle
                  res = not_found = -1

                  if not txt and not pattern:
                      return 0

                  if not pattern:
                      return 0

                  if len(pattern) > len(txt):
                      return not_found

                  lps = self.get_lps(pattern)
                  t = p = 0
                  while t < len(txt):
                      if txt[t] == pattern[p]:
                          t, p = t+1, p+1
                          if p == len(pattern):
                              res = t - len(pattern)
                              break
                      else:
                          if p == 0:
                              t += 1
                          else:
                              p =lps[p-1]

                  return res
            ```
    * 214:**Shortest** Palindrome (H)
       * Description:
         * Given a string s, you are **allowed to convert it to a palindrome by adding characters in front of it**. Find and return the shortest palindrome you can find by performing this transformation.
       * Approach1: Brute Force, Time:O(n^2), Space:O(n)
         * Reverse the string
         * Find the **longest common prefix of the str and suffix of reverse str**
         * For example:
           * str = "abade"
           * rev = "edaba"
           * the longest palindrom string would be rev + str = "edabaabade"
             * The longest common suffix of rev and prefix of str is aba
             * The shortest palindrom would be "edabaed"
         * Python
            ```python
            def shortestPalindrome(self, s: str) -> str:
              rev = s[::-1]
              n = len(s)
              # default is longest palindrom
              shortest_palindrom = rev + s
              for i in range(n):
                  suffix = rev[i:]
                  prefix = s[:n-i]
                  if prefix == suffix:
                      shortest_palindrom = rev[:i] + s
                      break

              return shortest_palindrom
            ```
       * Approach2: KMP, Time:O(n), Space:O(n)
         * The problem can be converted to longest palindrome substring **starts from 0**
         * For example: str = "abade"
             * The longest palindrom substr substring from 0 is "aba"
             * The **reverse** the substring after "aba" is "ed"
             * Append "ed" before the head, 'ed'abadede is the answer.
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
          * Python:
            ```python
            def get_lps(self, pattern):
              if len(pattern) == 1:
                  return [0]

              lps = [0] * len(pattern)
              p, s = 0, 1
              while s < len(lps):
                  if pattern[p] == pattern[s]:
                      p += 1
                      lps[s] = p
                      s += 1
                  else:
                      if p == 0:
                          """
                          do not match anything
                          lps[s] = 0
                          """
                          s += 1
                      else:
                        """
                        Reuse the prefix string that has been scanned.
                        The length of the longest common prefix suffix
                        are put in lps[p-1]
                        """
                        p = lps[p-1]

              return lps

            def shortestPalindrome(self, s: str) -> str:
              rev = s[::-1]
              lps = self.get_lps(f'{s}#{rev}')
              """
              The longest suffix of rev and prefix of s is lps[-1]
              """
              shortest_palindrom = rev[:len(s)-lps[-1]] + s
              return shortest_palindrom
  * **Palindrome**
     * 009: Palindrome **Number** (E)
       * notice the negative value:
         * example: -121
         * From left to right, it reads -121. From right to left, it becomes 121-.
       * Approach1: Covert to string, Time:O(n), Space:O(n)
         * Python
          ```python
          def isPalindrome(self, x: int) -> bool:
            s = str(x)
            l, r = 0, len(s)-1

            is_palindrom = True
            while l < r:
                if s[l] != s[r]:
                    is_palindrom = False
                    break

                l, r = l+1, r-1

            return is_palindrom
              ```
        * Approach2: Reverse the interger, Time:O(log(n)), Space:O(1)
          * Python
            ```python
            def isPalindrome(self, x: int) -> bool:
              if x < 0:
                  return False

              ori = x
              rev = 0
              while x:
                  pop = x % 10
                  x = x // 10
                  rev = rev * 10 + pop

              return rev == ori
            ```
     * 125:	Valid Palindrome (E)
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
         * Python
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
     * 409: Longest Palindrome (E)
       * Description:
         * Given a string which consists of lowercase or uppercase letters, **find the length of the longest palindromes that can be built with those letters**.
         * This is case sensitive, for example "Aa" is not considered a palindrome here.
       * Approach1: Use Counter, Time:O(n), Space:O(n)
         * Same idea like 266
         * Python
            ```python
            def longestPalindrome(self, s: str) -> int:
                counter = collections.Counter(s)
                longest_len = 0
                for cnt in counter.values():
                    if not longest_len % 2 and cnt % 2:
                      longest_len += 1

                    longest_len += (cnt // 2) * 2

                return longest_len
            ```
     * 267:	Palindrome **Permutation** II (M)
       * See backtracking
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
             * one character
             * memo(i, i) = true
           * case2: For i+1 == j
             * two characters
             * memo(i, i+1) = (s(i) == s(i+1))
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
                  # skip case 1 and case2
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
         * Python
            ```python
            def longestPalindrome(self, s: str) -> str:
              def expand_center(l, r):
                  while l >= 0 and r < len(s):
                      if s[l] != s[r]:
                          break
                      l, r = l-1, r+1

                  l, r = l+1, r-1
                  return (r-l+1), l, r

              res = [float('-inf'), None, None]
              for center in range(len(s)):
                  length, l, r = expand_center(center, center)
                  if length > res[0]:
                      res[0], res[1], res[2] = length, l, r

                  length, l, r = expand_center(center, center+1)
                  if length > res[0]:
                      res[0], res[1], res[2] = length, l, r

              return "" if res[0] == float('-inf') else s[res[1]:res[2]+1]
              ```
       * Approach4: Manacher's Algorithm, Time: O(n), Space: O(n)
         * Ref:
           * https://www.youtube.com/watch?v=nbTSfrEfo6M
     * 214:	**Shortest** Palindrome (H)
       * See KMP
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
       * Description:
         * Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
       * Valid Cases:
         * [] () {}
         * { [ () ] }
       * Invalid Cases:
         * { [ () ]
         * ))
         * ((
       * Approach1: Use stack, Time:O(n), Space:O(n)
         * Python
           ```python
           def isValid(self, s: str) -> bool:
              BRACKET_MAP = {'(': ')', '[': ']', '{': '}'}
              is_valid = True
              stack = []
              for c in s:
                  if c in BRACKET_MAP:
                      stack.append(c)
                  else:
                      if not stack or c != BRACKET_MAP[stack.pop()]:
                        is_valid = False
                        break

              return is_valid and len(stack) == 0
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
      * Description:
        * Check if the pattern is a subsequence of txt
      * Approach1: Iterate pattern, Time:O(n), Space:O(1)
        * Python
          ```python
          def isSubsequence(self, s: str, t: str) -> bool:
            pattern, txt = s, t

            if not pattern:
                return True

            is_subsequence = True
            j = 0
            for c in pattern:
                while j < len(txt) and txt[j] != c:
                    j += 1

                if j == len(txt):
                    is_subsequence = False
                    break
                else:
                    j += 1

            return is_subsequence
          ```
      * Approach2: Iterate txt, Time:O(n), Space:O(1)
        * Python
          ```python
          def isSubsequence(self, s: str, t: str) -> bool:
            pattern, txt = s, t

            if not pattern:
                return True

            is_subseq = False
            i = 0
            for c in txt:
                if pattern[i] == c:
                    i += 1
                    if i == len(pattern):
                        is_subseq = True
                        break

            return is_subseq
          ```
    * 1143: Longest Common Subsequence (M)
      * See DP
    * 674: Longest Continuous Increasing Subsequence (**LCIS**) (E)
      * See DP
    * 300: Longest Increasing Subsequence (**LIS**) (M)
      * See Binary Search
    * 792: Number of Matching Subsequences
    * 187: Repeated DNA Sequences (M)
    * 115: Distinct Subsequences (H)
  * **Reorder**
     * Ref:
       * https://docs.python.org/3/library/stdtypes.html
       * isspace, isalpha, isascii(3.7)
     * 344: Reverse String (E)
       * Approach1: Time:O(n), Space:O()
       * Python
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
       * Description:
         * You need to **reverse the first k characters for every 2k characters** counting from the start of the string.
         * If there are less than k characters left, reverse all of them.
         * example
           * input: s = "abcdefgh", k = 3
           * output: "cbadefhg"
       * Approach1: Scan From Left to right, Time:O(n), Space:O(1)
         * Python
            ```python
            def reverseStr(self, s: str, k: int) -> str:
              def reverse(s, l, r):
                  while l < r:
                      s[l], s[r] = s[r], s[l]
                      l, r = l+1, r-1

              if not s or k == 1:
                  return s

              s = list(s)
              left = right = 0
              rev = True
              for right in range(len(s)):
                  if (right + 1) % k == 0:
                      if rev:
                          reverse(s, left, right)
                      rev = not rev
                      left = right + 1

              if left < len(s) and rev:
                  reverse(s, left, len(s)-1)

              return ''.join(s)
            ```
       * Approach2: Jump, Time:O(n), Space:O(1)
         * Python
           ```python
            def reverseStr(self, s: str, k: int) -> str:
              def reverse(s, l, r):
                while l < r:
                    s[l], s[r] = s[r], s[l]
                    l, r = l+1, r-1

              if not s or k == 1:
                  return s

              s = list(s)
              left = 0
              n = len(s)
              while left < n:
                  right = min(n-1, left + k - 1)
                  reverse(s, left, right)

                  left += 2*k

              return ''.join(s)
           ```
     * 151:	Reverse **Words** in a String	(M)
       * Description:
         * Given an input string, reverse the string word by word.
         * 151 has many corners cases.
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
         * Python
         ```python
        def reverseWords(self, s: List[str]) -> None:

          def reverse(s, l, r):
              while l < r:
                  s[l], s[r] = s[r], s[l]
                  l, r = l+1, r-1

          n = len(s)
          reverse(s, 0, n-1)
          w_s = 0
          while w_s < n:
              while w_s < n and s[w_s].isspace():
                  w_s += 1

              # can not find the start position of the word
              if w_s == n:
                  break

              w_e = w_s
              while w_e+1 < n and not s[w_e+1].isspace():
                  w_e += 1

              reverse(s, w_s, w_e)

              w_s = w_e + 2
         ```
     * 557: Reverse **Words** in a String III (E)
       * Simpler than 186
       * Description:
         * example:
           * Input: "Let's take LeetCode contest"
           * Output: "s'teL ekat edoCteeL tsetnoc"
       * Approach1:
         * Python
          ```python
          def reverseWords(self, s: str) -> str:
            def reverse(s, l, r):
                while l < r:
                    s[l], s[r] = s[r], s[l]
                    l, r = l+1, r-1

            if not s:
                return s

            s = list(s)
            n = len(s)
            w_s = 0
            while w_s < n:

                while w_s < n and s[w_s].isspace():
                    w_s += 1

                if w_s == n:
                    break

                w_e = w_s
                while w_e + 1 < n and not s[w_e+1].isspace():
                    w_e += 1

                reverse(s, w_s, w_e)
                w_s = w_e + 1

          return ''.join(s)
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
     * 917: Reverse Only Letters(E)
       * The same like 345
       * Description:
       * Python
        ```python
        def reverseOnlyLetters(self, S: str) -> str:
          if not S:
              return S

          s = list(S)
          n = len(s)
          l, r = 0, n-1

          while l < r:
              while l < r and not s[l].isalpha():
                  l += 1

              while l < r and not s[r].isalpha():
                  r -= 1

              if l >= r:
                  break

              s[l], s[r] = s[r], s[l]
              l, r = l+1, r-1

          return ''.join(s)
        ```
     * 058: Length of Last Word (E)
       * Approach1: Seach **from the end to the beginning**.
         * Python
            ```python
            def lengthOfLastWord(self, s: str) -> int:
              cnt = 0
              for i in range(len(s)-1, -1, -1):
                  if s[i].isspace():
                      if cnt:
                          break
                  else:
                      cnt += 1

              return cnt
            ```
       * Approach2: Two pointers
        * Python
          ```python
          def lengthOfLastWord(self, s: str) -> int:
            n = len(s)
            w_start = n - 1
            cnt = 0
            boundary = -1
            while w_start >= 0:
                while w_start > boundary and s[w_start].isspace():
                    w_start -= 1

                if w_start == boundary:
                    break

                w_end = w_start

                while w_end - 1 > boundary and not s[w_end-1].isspace():
                    w_end -= 1

                cnt = w_start - w_end + 1
                break

            return cnt
        ```
     * 358: Rearrange String k Distance Apart (H)
  * Encode(Compression) and Decode:
    * 038: Count and Say (E)
      * Description:
        ```txt
         1.     1
         2.     11
         3.     21
         4.     1211
         5.     111221
         6.     312211
        ```
      * Approach1: Two Pointers
        * Python
          ```python
          def countAndSay(self, n: int) -> str:
            cur = ['1']
            for _ in range(2, n+1):
                nxt = []
                start = 0
                while start < len(cur):
                    end = start
                    while end + 1 < len(cur) and cur[end+1] == cur[start]:
                        end += 1

                    cnt = end - start + 1
                    nxt.append(str(cnt))
                    nxt.append(cur[start])
                    start = end + 1

                cur = nxt

            return ''.join(cur)
          ```
      * Approach2: Cnt and One Pointer
        * Python
          ```python
          def countAndSay(self, n: int) -> str:
            cur = ['1']
            for _ in range(2, n+1):
                nxt = []
                end = 0
                while end < len(cur):
                    cnt = 1
                    while end + 1 < len(cur) and cur[end+1] == cur[end]:
                        cnt, end = cnt + 1, end + 1

                    nxt.append(str(cnt))
                    nxt.append(cur[end])
                    end += 1

                cur = nxt

            return ''.join(cur)
          ```
    * 443: String Compression (E)
      * Description:
        * The length after compression must always be smaller than or equal to the original array.
        * Example:
          * 1:
            * Intput: ["a","a","b","b","c","c","c"]
            * Output: ["a","3","b","2","c","3"]
          * 2
            * Input:  [a]
            * Output: [a]
          * 3:
            * Input: ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
            * Output:  ["a", "1", "2"]
          * 4:
            * Input: ["a","a","a","a","a","b"]
            * Output: ["a","5", "b"]
      * Approach1: Time:O(n), Space:O(1)
        * Python
          ```python
          def compress(self, chars: List[str]) -> int:
            border = 0
            end = 0
            while end < len(chars):
                cnt = 1
                while end + 1 < len(chars) and chars[end + 1] == chars[end]:
                    cnt, end = cnt+1, end+1

                chars[border] = chars[end]
                border += 1
                if cnt > 1:
                    """
                    Input: ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
                    Output would be ["a", "1", "2"] rather than ["a", "12"]
                    """
                    for c in str(cnt):
                        chars[border] = c
                        border += 1

                end += 1

            return border
          ```
    * 394: Decode String (M)
      * See Queue and Stack Section
    * 271: Encode and Decode Strings (M)
      * Approach1: Non-ASCII Delimiter
      * Approach2: Chunked Transfer Encoding
  * **Isomorphism and Pattern**:
     * 205: **Isomorphic** Strings (E)
        * Description:
          * Two strings are isomorphic if the characters in s can be replaced to get t.
          * All occurrences of a character must be replaced with another character **while preserving the order of characters.**
          * Example:
            * aabbaa,  112233 -> False
            * aabbaa,  112211 -> True
        * Approach1, Use Dict to track chr positions
          * Use hash Table to keep **last seen index**
          * Python
            ```python
            def isIsomorphic(self, s: str, t: str) -> bool:
              if len(s) != len(t):
                  return False

              is_isomorphic = True

              f = lambda: -1
              d1 = collections.defaultdict(f)
              d2 = collections.defaultdict(f)

              for i, (c1, c2) in enumerate(zip(s, t)):
                  if d1[c1] != d2[c2]:
                      is_isomorphic = False
                      break

                  d1[c1] = d2[c2] = i

              return is_isomorphic
            ```
     * 290: Word **Pattern** (E)
       * The same concept as 205
       * Approach1, Use Hash Table
          * Use hash Table to keep **last seen index**
       * Python
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
     * 291: Word Pattern II (H)
  * **Anagram**
    * The key is how to calculate **signatures**.
     * 242: Valid Anagram (E)
       * Description:
         * Input:
           * s = "anagram", t = "nagaram"
         * Output: True
       * Approach1: Use hash table, Time:O(n), Space:O(n)
         * Python Solution
            ```python
            def isAnagram(self, s: str, t: str) -> bool:
              if len(s) != len(t):
                  return False

              is_anagram = True
              s_counter = collections.Counter(s)
              for c in t:
                  if s_counter[c] == 0:
                      is_anagram = False
                      break

                  s_counter[c] -= 1

              return is_anagram
          ```
       * Approach2: Use sort, Time:O(nlogn), Space:O(n)
         * Python
          ```python
          def isAnagram(self, s: str, t: str) -> bool:
            if len(s) != len(t):
                return False

            return sorted(s) == sorted(t)
          ```
     * 049: Group Anagrams (M)
       * Description:
         * Input:
           * ["eat", "tea", "tan", "ate", "nat", "bat"]
         * Output:
           * [ ["ate","eat","tea"],
               ["nat","tan"],
               ["bat"]
            ]
       * Approach1: Categorized by **sorted string**, Time: O(n*klog(k)) Space: O(nk)
         * Python
            ```python
            def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
              d = collections.defaultdict(list)
              for s in strs:
                d[tuple(sorted(s))].append(s)
            return d.values()
            ```
       * Approach2: Categorized by **Character Count**, Time: O(nk), Space: O(nk)
         * Python
            ```python
            def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
              def get_sig(s):
                  sig = [0] * 26
                  for c in s:
                      sig[ord(c) - ORD_A] += 1
                  return str(sig)

              ORD_A = ord('a')
              d = collections.defaultdict(list)
              for s in strs:
                  d[get_sig(s)].append(s)

              return d.values()
            ```
     * 249: Group **Shifted Strings** (M)
       * Description:
         * "abc" -> "bcd" -> ... -> "xyz"
         * Input:
           * ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
         * Output:
           * [
              ["abc","bcd","xyz"],
              ["az","ba"],
              ["acef"],
              ["a","z"]
           * ]
       * Approach1: Group by diff string, Time:O(nk)
         * Example:
           * [a, b] and [z, a]
             * ord(a) = 97
             * ord(b) = 98
             * ord(z) = 122
           * (ord(b) - ord(a)) ≡ 1 (mod 26)
           * (ord(a) - ord(z)) ≡ -25 ≡ 1 (mod 26)
           * 1 is **congruent** to 25 (modulo 26)
       * Python
         ```python
         def groupStrings(self, strings: List[str]) -> List[List[str]]:
            def get_sig(s):
              ord_first = ord(s[0])
              return tuple((ord(c)- ord_first)%26  for c in s)

            d = collections.defaultdict(list)
            for s in strings:
                d[get_sig(s)].append(s)

            return d.values()
         ```
    * 848: Shifting Letters (M)
      * Approach1: shift array, Time:O(n), Space:O(n)
        * Python
          ```python
          def shiftingLetters(self, S: str, shifts: List[int]) -> str:
            if not S:
                return None

            if not shifts:
                return S

            ORD_A = ord('a')
            array = list(S)

            shift_memo = [0] * len(shifts)
            for i, shift in enumerate(shifts):
                shift_memo[0] += shift
                if i + 1 < len(shift_memo):
                    shift_memo[i+1] -= shift

            cur_shift = 0
            for i, shift in enumerate(shift_memo):
                cur_shift += shift
                shift_memo[i] = cur_shift

            for i, shift in enumerate(shift_memo):
                c = array[i]
                # ord(c)-ORD_A is the base shift
                array[i] = chr(ORD_A + (ord(c)-ORD_A + shift)%26)

            return "".join(array)
          ```
      * Approach2: maintain a shift variable, Time:O(n), Space:O(1)
        * Python
          ```python
          def shiftingLetters(self, S: str, shifts: List[int]) -> str:
            if not S:
                return None

            if not shifts:
                return S

            ORD_A = ord('a')
            array = list(S)

            cur_shifts = sum(shifts)
            for i, shift in enumerate(shifts):
                c = array[i]
                # ord(c)-ORD_A is the base shift
                array[i] = chr(ORD_A + (ord(c)-ORD_A + cur_shifts)%26)
                cur_shifts -= shift

            return "".join(array)
          ```
  * Deduction:
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
      * see backtracking
  * Counter:
    * 387: First Unique Character in a String (E)
       * Approach1: Use char counter, Time: O(n), Space: O(c)
         * Python
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
       * Approach2: Sorting, Time:O(nlogn), Space:O(1)
    * 383: Ransom Note (E)
      * Description:
        * write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false.
        * Each letter in the magazine string can only be used once in your ransom note.
      * Approach1: Counter, Time:O(m+n), Space:O(m)
        * Python
        ```python
          def canConstruct(self, ransomNote: str, magazine: str) -> bool:
            counter = collections.Counter(magazine)
            can_construct = True
            for c in ransomNote:
                if c not in counter or counter[c] <= 0:
                    can_construct = False
                    break

                counter[c] -= 1

            return can_construct
        ```
  * Other
     * 014: Longest **Common Prefix** (E)
       * Approach1: **vertical scanning**, Time: O(mn), Space: O(1)
         * Notice the edge cases:
           * 1 strings
           * 1 character for each string, [a, b]
         * Time: O(mn)
           * Where m is the minimum length of str in strs and n is the number of strings.
         * Python
            ```python
            def longestCommonPrefix(self, strs: List[str]) -> str:
              if len(strs) == 0:
                  return ""

              """
              vertical scanning
              the lognest common prefix is strs[0]
              """
              prefix = strs[0]
              border = len(prefix)
              for i in range(len(prefix)):
                  for j in range(1, len(strs)):
                      s = strs[j]
                      if i == len(s) or prefix[i] != s[i]:
                          border = i
                          break
                  else:
                      continue
                  break

              return prefix[:border]
            ```
     * 087: Scramble String (H)
### Array
  * **Check Duplicate**
    * 217: Contains Duplicate (E)
      * Description:
        * Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
      * Approach1: Use Counter, Time:O(n), Space:O(n)
        * Python
          ```python
          def containsDuplicate(self, nums: List[int]) -> bool:
            d = dict()
            res = False
            for n in nums:
                if n in d:
                  res = True
                  break

                d[n] = True
            return res
          ```
      * Approach2: Sorting, Time:O(nlogn), Space:O(logn~n)
        * Python
          ```python
          def containsDuplicate(self, nums: List[int]) -> bool:
            nums.sort()
            res = False
            for i in range(0, len(nums)-1):
                if nums[i] == nums[i+1]:
                    res = True
                    break
            return res
          ```
    * 219: Contains Duplicate II (E)
      * Description:
        * Find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between **i and j is at most k**.
      * Approach1: Use dict to store index.
        * Python
        ```python
          def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
            if not nums:
                return False

            d = dict()
            res = False
            for idx, n in enumerate(nums):
                if n in d and idx - d[n] <= k:
                    res = True
                    break

                d[n] = idx

            return res
          ```
    * 220: Contains Duplicate III (M)
      * Description:
        * find out whether there are two distinct indices i and j in the array such that
          * The **absolute difference** between **nums[i] and nums[j] is at most t**.
          * The **absolute difference** between **i and j is at most k**.
      * Approach1: Brute Force:O(n^2), Space:O(1)
      * Approach2: Buckets: O(n), Space:O(n)
        * Ref:
          * https://leetcode.com/problems/contains-duplicate-iii/discuss/61639/JavaPython-one-pass-solution-O(n)-time-O(n)-space-using-buckets
          * The difference of values in the same bucket <= t
          * keep at most k buckets
        * Python
          ```python
          def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
            if t < 0:
                return False

            """
            The difference of values in the same bucket <= t
            """
            buckets = dict()
            bucket_size = t + 1
            res = False
            for i, n in enumerate(nums):
                b_idx = n // bucket_size

                # check the same bucket and neighbor buckets
                if b_idx in buckets \
                  or b_idx - 1 in buckets and n - buckets[b_idx - 1] <= t \
                  or b_idx + 1 in buckets and buckets[b_idx + 1] - n <= t:
                    res = True
                    break

                buckets[b_idx] = n
                if i >= k:
                    buckets.pop(nums[i - k] // bucket_size)

            return res
          ```
  * **Remove Duplicate**
    * Tips:
      * The key is how to handle the border
    * 027: Remove elements (E)
      * Description:
        * Given an array nums and a value val, remove all instances of that value in-place and return the new length.
      * Approach1:
        * Like partition step of quick sort (keep the border)
        * Python
          ```python
          def removeElement(self, nums: List[int], val: int) -> int:
            border = 0
            for n in nums:
                if n != val:
                    nums[border] = n
                    border += 1

            return border
          ```
    * 026: Remove **Duplicates** from **Sorted Array** (E)
      * Description:
        * Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
      * Python
        ```python
        def removeDuplicates(self, nums: List[int]) -> int:
          if not nums:
            return 0
          i = 0
          for j in range(1, len(nums)):
              if nums[j] != nums[i]:
                  i += 1
                  nums[i] = nums[j]
          return i + 1
        ```
    * 080: Remove **Duplicates** from **Sorted Array** II (M) *
      * Description:
        * Given a sorted array nums, remove the duplicates in-place such that duplicates appeared **at most twice** and return the new length.
        * example:
          * input: [1,1,1,2,2,3], output: [1,1,2,2,3]
          * input: [1,1,1,2,2,3], output: [1,1,2,2,3]
      * Approach1: border + cnt, Time:O(n), Space:O(1)
        * Python:
          ```python
          def removeDuplicates(self, nums: List[int]) -> int:
            if len(nums) <= 2:
                return len(nums)

            max_duplicate = 2
            border = 0
            cnt = 1
            for i in range(1, len(nums)):
                if nums[i] == nums[border]:
                    cnt += 1
                    if cnt <= max_duplicate:
                        border += 1
                        nums[border] = nums[i]

                else:
                    border += 1
                    nums[border] = nums[i]
                    cnt = 1

            return border + 1
          ```
      * Approach2: border only, Time:O(n), Space:O(1)
        * Python
          ```python
          def removeDuplicates(self, nums: List[int]) -> int:
            if len(nums) <= 2:
                return len(nums)

            border = 1
            for i in range(2, len(nums)):
                if nums[i] > nums[border-1]:
                    border += 1
                    nums[border] = nums[i]

            return border + 1
          ```
  * **Containers**
    * 011: Container With Most Water (M)
      * How to calculate the area?
        * min(left_border, right_border) * width
      * Approach1: brute force, Time: O(n^2), Space: O(1)
        * Calculating area for all height pairs.
        * Python:
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
          * Python
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
    * 042: Trapping Rain Water (H)
      * Description:
        * Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
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
          * Python
            ```python
            def trap(self, height: List[int]) -> int:
              if len(height) <= 2:
                return 0

              n = len(height)

              left_max = [0] * n
              left_max[0] = height[0]
              for i in range(1, n):
                  """
                  left_max[i] should consider height[i]
                  since if height[i] is the new border, the area of ith bin should be 0
                  example:
                  [2, 10, 5], for 1th bin, if we do not inlcude 10, the area would be negative.
                  """
                  left_max[i] = max(left_max[i-1], height[i])

              right_max = [0] * n
              right_max[n-1] = height[n-1]
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
          * Python
            ```python
            def trap(self, height: List[int]) -> int:
              if len(height) <= 2:
                  return 0

              area = 0
              left, right = 0, len(height)-1
              max_left = max_right = 0

              while left <= right:
                  # area depends on left border
                  if height[left] <= height[right]:
                      if height[left] >= max_left:
                          """
                          new left border, do not need to calculate area
                          """
                          max_left = height[left]
                      else:
                          area += (max_left - height[left])
                      left += 1

                  # area depends on right border
                  else:
                      if height[right] >= max_right:
                          """
                          new right border, do not need to calculate area
                          """
                          max_right = height[right]
                      else:
                          area += (max_right - height[right])
                      right -= 1

              return area
            ```
    * 407: Trapping Rain Water II (H)
  * **Jump Game**:
    * 055: Jump Game (M)
      * See DP
  * **H-Index**
    * Description:
      * According to the definition of h-index on Wikipedia: "**A scientist has index h if h of his/her N papers have at least h citations each**, and the other N − h papers have no more than h citations each."
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
    * 275: H-Index II (M)
      * Please refer binary search
  * **Best Time to Buy and Sell Stock**
    * See DP
  * **Shortest Word Distance**
    * 243: Shortest Word Distance (E)
       * Description:
         * Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.
         * You may assume that **word1 does not equal to word2**.
         * word1 and word2 are both in the list
       * Approach1: Time:O(n), Space:O(1)
         * Calculate the distance and update the shortest distance in each round.
           * Draw some cases1:
             * 1 2 1* -> will cover two cases 1-2 and 2-1*
             * 1 2 2* 1* -> still can cover
               * 1-2* will not cover, but it is bigger than 1-2
               * 2-1* is the same, it is bigger than 2*-1*
           * Python
              ```python
              def shortestDistance(self, words: List[str], word1: str, word2: str) -> int:
                idx1 = idx2 = not_found = -1
                min_dis = len(words)

                for idx, w in enumerate(words):
                    if w == word1:
                        idx1 = idx
                    elif w == word2:
                        idx2 = idx

                    if idx1 != not_found and idx2 != not_found:
                        min_dis = min(min_dis, abs(idx1-idx2))

                return min_dis
              ```
    * 245: Shortest Word Distance III (M)
      * Description
        * Allow **duplicated words**.
      * Approach1:
        * The same concept with 243, but need to handle duplicated words
        * Python
            ```python
            def shortestWordDistance(self, words: List[str], word1: str, word2: str) -> int:
              idx1 = idx2 = not_found = -1
              min_dis = len(words)

              same = True if word1 == word2 else False

              for idx, w in enumerate(words):
                  if w == word1:
                      if not same:
                          idx1 = idx
                      else:
                          idx1, idx2 = idx2, idx

                  elif w == word2:
                      idx2 = idx

                  if idx1 != not_found and idx2 != not_found:
                      min_dis = min(min_dis, abs(idx1-idx2))

              return min_dis
            ```
    * 244: Shortest Word Distance II (M)
       * Description:
         * **Init once** and **search for multiple time**.
       * Approach1, indexes list of words, Time:O(K + L), Space:O(n)
         * Using **Preprocessed Sorted Indices** and two pointers to traverse
           * Space: O(n)
             * For or the dictionary that we prepare in the constructor.
             * The keys represent all the unique words in the input and the values represent all of the indices from 0 ... N0...N.
         * Complexity:
           * Init:
             * O(n), where n is the number of words.
           * Find:
           * O(K + L), where K and L represent the number of occurrences of the two words.
         * Python
            ```python
            class WordDistance:
              def __init__(self, words: List[str]):
                  self.word_d = collections.defaultdict(list)
                  for idx, w in enumerate(words):
                      self.word_d[w].append(idx)

              def shortest(self, word1: str, word2: str) -> int:
                  idxes1 = self.word_d[word1]
                  idxes2 = self.word_d[word2]

                  i = j = 0
                  min_dis = float('inf')
                  while i < len(idxes1) and j < len(idxes2):
                      idx1, idx2 = idxes1[i], idxes2[j]
                      min_dis = min(min_dis, abs(idx1-idx2))
                      if idx1 <= idx2:
                          i += 1
                      else:
                          j += 1

                  return min_dis
  * **Interval**
    * 252: Meeting Rooms (E)
      * Description:
        * Check if one person **can attend all meetings**.
      * FAQ
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
        * Python
          ```python
          def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
            START, END = 0, 1
            can_attend_all = True

            intervals.sort(key=lambda interval: interval[START])

            for i in range(1, len(intervals)):
                if intervals[i-1][END] > intervals[i][START]:
                    can_attend_all = False

            return can_attend_all
          ```
    * 253: Meeting Rooms II (M)
      * see heap (Priority Queue) section
    * 056: Merge Intervals (M)
      * Description:
        * Given a collection of intervals, merge all overlapping intervals.
        * example:
          * Input: [[1,3],[2,6],[8,10],[15,18]], Output: [[1,3],[2,6],[8,10],[15,18]]
      * FAQ:
        * For greedy method, why we don't need a max-heap to track?
          * Because merged_interval[-1][END] is the maximum end value in the list
      * 3 cases:
        * case1: no overloap
        * case2 &case3: having overlap, merge two intervals
        * ```txt

          case1:  [-----]
                          [-----]

          case2: [-----------------]
                          [-----]

          case3: [------------]
                          [--------]

          ```
      * Approach1: Brute Force, Time:O(n^2)
      * Approach2: Greedy, Time: O(nlogn), Space: O(n)
        * Sort the intervals by start time,
        * Update the output if there exists interval overlap.
        * Python
            ```python
            def merge(self, intervals: List[List[int]]) -> List[List[int]]:
              if not intervals:
                  return []

              START, END = 0, 1

              intervals.sort(key=lambda interval: interval[START])
              merged_intervals = [intervals[0][:]]

              for i in range(1, len(intervals)):
                  last = merged_intervals[-1]
                  nxt = intervals[i]

                  if last[END] >= nxt[START]:
                      # merge
                      last[END] = max(last[END], nxt[END])
                  else:
                      merged_intervals.append(nxt[:])

              return merged_intervals
            ```
    * 435: Non-overlapping Intervals (M)
      * Description:
        * Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
      * FAQ:
        * For greedy method, why we don't need a max-heap to track?
          * Use a prev_end variable can do the same thing
          * similar to the problem 56
        * 3 cases:
          * case1: no overloap
          * case2: remove interval[i]
          * case3: remove interval[i+1]
        * ```txt

          case1:  [-----]
                          [-----]

          case2: [-----------------]
                          [-----]

          case3: [------------]
                          [--------]

         ```
      * Approach1: DP, Time:O(n^2), Space:O(n)
        * memo[i] stores the maximum number of valid intervals that can be included
        * Python
          ```python
          def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
            if not intervals:
                return 0

            START, END = 0, 1
            intervals.sort(key=lambda interval: interval[START])

            """
            stores the maximum number of valid intervals that can be included
            """
            memo = [1] * len(intervals)

            for cur in range(1, len(intervals)):
                max_valid = 0
                for prev in range(0, cur):
                    if intervals[prev][END] <= intervals[cur][START]:
                        max_valid = memo[prev]

                if max_valid:
                    memo[cur] = max_valid + 1

            """
            min remove cnt is total cnt minus max valid cnt
            """
            return len(intervals) - max(memo)
          ```
      * Approach2: Greedy, Time:O(nlong), Space:O(n)-
        * Python
          ```python
          def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
            if not intervals:
                return 0

            START, END = 0, 1
            intervals.sort(key=lambda interval: interval[START])

            remove_cnt = 0
            prev_end = intervals[0][END]

            for i in range(1, len(intervals)):
                nxt = intervals[i]
                """
                END == START is not overlap
                """
                if prev_end <= nxt[START]:
                    prev_end = nxt[END]
                else:
                    """
                    keep the interval with smaller end time,
                    that is, remove the interval with bigger end time
                    """
                    remove_cnt += 1
                    prev_end = min(prev_end, nxt[END])

            return remove_cnt
          ```
    * 452: Minimum Number of Arrows to Burst Balloons
      * Description:
        * For each balloon, provided input is the **start and end coordinates** of the **horizontal** diameter.
        * Since it's horizontal, y-coordinates don't matter and hence the x-coordinates of start and end of the diameter suffice.
        * An arrow can be shot up exactly vertically from different points along the x-axis.
          * A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend.
      * The problem is similar to 435
      * Approach1: Sorting by start, Time:O(nlong), Space:O(n)
        * Python
          ```python
          def findMinArrowShots(self, points: List[List[int]]) -> int:
            if not points:
                return 0

            START, END = 0, 1
            points.sort(key=lambda point: point[START])

            arrow = 1
            prev_end = points[0][END]

            for i in range(1, len(points)):
                cur = points[i]
                """
                can not merge
                """
                if prev_end < cur[START]:
                    prev_end = cur[END]
                    arrow += 1
                else:
                    """
                    after merged, keep the smaller prev_end
                    """
                    prev_end = min(prev_end, cur[END])

            return arrow
          ```
    * 057: Insert Interval (H)
    * 352: Data Stream as Disjoint Intervals (H)
  * **Subarray**
    * 053: Maximum Subarray (E)
      * Description
        * Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
      * Approach1: Brute Force, Time:O(n^2), Space:O(1)
         * Python
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
      * Approach2: DP with memo, Time:O(n), Space:O(n)
        * meomo[i], largest sum of contiguous subarray inclued nums[i]
        * Python
          ```python
          def maxSubArray(self, nums: List[int]) -> int:
          if not nums:
              return 0

          max_sum = nums[0]
          memo = [0] * len(nums)
          memo[0] = nums[0]

          for i in range(1, len(nums)):
              n = nums[i]
              memo[i] = max(memo[i-1] + n, n)
              max_sum = max(max_sum, memo[i])

          return max_sum
          ```
      * Approach3: Dp with variables, Time:O(n), Space:O(1)
         * Python
            ```python
            def maxSubArray(self, nums: List[int]) -> int:
              if not nums:
                  return 0

              cur_sum = max_sum = nums[0]
              for i in range(1, len(nums)):
                  cur_sum = max(cur_sum+nums[i], nums[i])
                  max_sum = max(max_sum, cur_sum)

              return max_sum
            ```
    * 152: Maximum **Product** Subarray (M)
      * Description:
        * Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.
      * The challenge is how to handle negative values.
      * Approach1: Brute Force, Time:O(n^2)
      * Approach2: DP with memo, Time: O(n), Space: O(n)
        * Python
          ```python
          def maxProduct(self, nums: List[int]) -> int:
            MAX, MIN = 0, 1
            memo = [[0, 0] for i in range(len(nums))]
            max_product = memo[0][MAX] = memo[0][MIN] = nums[0]

            for i in range(1, len(nums)):
                n = nums[i]
                prev_max, prev_min = memo[i-1][MAX], memo[i-1][MIN]
                if n < 0:
                    prev_max, prev_min = prev_min, prev_max

                memo[i][MAX] = max(prev_max * n, n)
                memo[i][MIN] = max(prev_min * n, n)

                max_product = max(max_product, memo[i][MAX])

            return max_product
          ```
      * Approach3: DP with variables, Time: O(n), Space: O(1)
        * The concept is just like 53: maximum subarray, but need to notice negative value in the array.
        * Python
          ```python
          def maxProduct(self, nums: List[int]) -> int:
            max_product = cur_max = cur_min = nums[0]

            for i in range(1, len(nums)):
                n = nums[i]

                if n < 0:
                    cur_max, cur_min = cur_min, cur_max

                cur_max = max(n, cur_max*n)
                cur_min = max(n, cur_min*n)
                max_product = max(max_product, cur_max)

            return max_product
          ```
    * 238: **Product** of Array **Except Self** (M)
      * Description:
        * example:
          * Input: [1,2,3,4], Output: [24,12,8,6]
      * Approach1: Allow to use Division, Time:O(n), Space:O(1)
        * We can simply take the product of all the elements in the given array and then, for each of the elements xx of the array, we can simply find product of array except self value by dividing the product by xx.
        * Containing zero cases
          * exactly 1 zero
          * more than 1 zero
        * Python
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
              for idx, n in enumerate(nums):
                output[idx] = product // n
            return output
            ```
      * Approach2: Not Allow to use Division dp-1: Time:O(n), Space:O(n)
        * **For every given index i, we will make use of the product of all the numbers to the left of it and multiply it by the product of all the numbers to the right**.
        * Python:
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

          for i, (l, r) in enumerate(zip(left, right)):
              output[i] = l * r

          return output
        ```
      * Approach3: Not Allow to use Division dp-2: Time:O(n), Space:O(n)
          * We can use a variable to replace right_product array
          * Python:
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
      * Description
        * Given a sorted integer array without duplicates, return the summary of its ranges.
        * Input:
          * [0,1,2,4,5,7]
        * Output:
          * ["0->2","4->5","7"]
      * Approach1: Time:O(n), Space:O(1)
        * Python
          ```python
          def summaryRanges(self, nums: List[int]) -> List[str]:
              def append_output(start, end):
                if start == end:
                    output.append(str(nums[start]))
                else:
                    output.append(f'{nums[start]}->{nums[end]}')

            n = len(nums)
            output = list()
            start = 0
            while start < n:
                end = start
                while end + 1 < n and nums[end]+1 == nums[end+1]:
                    end += 1

                append_output(start, end)

                start = end + 1

            return output
          ```
      * Approach2: Change input, Time:O(n), Space:O(1)
        * This is not a good paractice
        * Python
          ```python
          def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
            def format_output(left_b, right_b):
                if right_b - left_b <= 1:
                    return
                elif right_b - left_b == 2:
                    output.append(str(left_b+1))
                else:
                    output.append(f'{left_b+1}->{right_b-1}')

            left_b = lower - 1
            output = []
            nums.append(upper+1)
            for n in nums:
                right_b = n
                format_output(left_b, right_b)
                left_b = right_b

            return output
          ```
    * 163: **Missing Ranges** (M)
      * Description
        * Given a sorted integer array nums, where the range of elements are in the inclusive range [lower, upper], return its missing ranges.
        * Example:
          * Input: nums = [0, 1, 3, 50, 75], lower = 0 and upper = 99
          * Output: ["2", "4->49", "51->74", "76->99"]
      * Approach1: Time:O(n), Space:O(1)
        * Need to know how to handle the boundary properly.
        * Python
          ```python
          def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
            def append_output(left_b, right_b):
                diff = right_b - left_b
                if diff == 2:
                    output.append(str(left_b+1))
                elif diff > 2:
                    output.append(f"{left_b+1}->{right_b-1}")

            output = []
            left_b = lower - 1

            for n in nums:
                right_b = n
                append_output(left_b, right_b)
                left_b = right_b

            right_b = upper + 1
            append_output(left_b, right_b)

            return outputt(left_boundary, right_boundary, output)
            return output
          ```
  * Prefix Sum:
    * 303: Range Sum Query - Immutable (E)
      * Approach1: Prefix sum
        * Python
          ```python
          class NumArray:
            def __init__(self, nums: List[int]):
                self.n = len(nums)
                self.d = {-1: 0}
                prefix = 0
                for idx, n in enumerate(nums):
                    prefix += n
                    self.d[idx] = prefix

            def sumRange(self, i: int, j: int) -> int:
                if i < 0 or j > self.n-1 or i > j:
                    return None

                # prefix[start] + target = prefix[end]
                start = i-1
                end = j
                target = self.d[end] - self.d[start]
                return target
          ```
    * 325: Maximum Size Subarray Sum **Equals k** (M)
      * Description
        * Find the maximum length of a subarray that sums to k
      * Approach1: Brute force, O(n^2), Space: O(1)
        * List all pairs of acc and keep the max
        * Python
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
      * Approach2: prefix sum, Time: O(n), Space: O(n)
        * [Concept](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/discuss/77784/O(n)-super-clean-9-line-Java-solution-with-HashMap)
          * Use hash table
            * key: accumulation value
            * val: index
        * Use hash table to keep acc value and the index.
        * Python
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
      * Approach2: Use dict, Time: O(n), Space: O(n)
        * Use hash table to keep acc value and cnt
        * Python
          ```python
          def subarraySum(self, nums: List[int], k: int) -> int:
            acc = cnt = 0
            d = collections.defaultdict(int)
            """
            init value for acc == k (starting from index 0)
            """
            d[0] += 1

            for n in nums:
              acc += n
              """
              target_acc + k = acc
              """
              target_acc = acc - k

              if target_acc in d:
                  cnt += d[target_acc]
              d[acc] += 1
            return cnt
          ```
    * 525: Contiguous Array (M)
      * Description:
        * Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
      * Approach1: Brute Force:O(n^2), Space:O(1)
      * Approach2: Count and indexes: Time:O(n), Space:O(n)
        * Ref:
          * https://leetcode.com/problems/contiguous-array/discuss/99655/Python-O(n)-Solution-with-Visual-Explanation
        * To find the maximum length, we need a dict to store the value of count (as the key) and its associated index (as the value). We only need to save a count value and its index at the first time, when the same count values appear again, we use the new index subtracting the old index to calculate the length of a subarray.
        * Python
          ```python
          def findMaxLength(self, nums: List[int]) -> int:
            if not nums:
                return 0

            # for subarray starting from index 0
            d = {0: -1}
            max_len = 0
            cnt = 0
            for idx, n in enumerate(nums):
                if n == 0:
                    cnt -= 1
                else:
                    cnt += 1

                if cnt in d:
                    # don't update d since we want to find max
                    max_len = max(max_len, idx-d[cnt])
                else:
                    d[cnt] = idx

            return max_len
          ```
  * **Order**
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
    * 896: Monotonic Array
      * Approach1: Time:O(n), Space:O(1)
        * Python
          ```python
          def isMonotonic(self, A: List[int]) -> bool:
            increase = decrease = True
            for i in range(len(A)-1):
                if A[i] > A[i+1]:
                    increase = False

                if A[i] < A[i+1]:
                    decrease = False

            return increase or decrease
          ```
  * Counter:
    * 299: Bulls and Cows (M)
      * Description:
        * Bull:
          * Two characters are the same and having the same index.
        * Cow
          * Two characters are the same but do not have the same index.
        * example:
          * Intput: secret = "1807", guess = "7810"
          * Output: "1A3B", 1 bull (8) and 3 cows (0,1,7)
      * Approach1: Hash TableTime O(n), Space O(n) and **one pass**
        * Use **hash Table** to count cows.
        * Python
          ```python
          def getHint(self, secret: str, guess: str) -> str:
            bull = cow = 0

            d = collections.defaultdict(int)
            for c1, c2 in zip(secret, guess):
                if c1 == c2:
                    bull += 1
                else:
                    if d[c1] < 0:
                        cow += 1

                    if d[c2] > 0:
                        cow += 1

                    d[c1] += 1
                    d[c2] -= 1

            return f'{bull}A{cow}B'
          ```
    * 1002: Find Common Characters
      * Description:
        * Given an array A of strings made only from lowercase letters, return a list of all characters that show up in all strings within the list (including duplicates).
      * Example:
        * Input: ["bella","label","roller"], Output: ["e","l","l"]
        * Input: ["cool","lock","cook"], Output: ["c","o"]
      * Approach1: Time:O(kn), Space:O(c)
        * k: number of string, n: avg length of string, c: signature size
        * Python
          ```python
          def commonChars(self, A: List[str]) -> List[str]:
            def get_counter(s):
                sig = [0 for _ in range(26)]
                for c in s:
                    sig[ord(c)-ord_a] += 1

                return sig

            if not A:
                return []

            n = len(A)
            ord_a = ord('a')
            min_counter = get_counter(A[0])
            for i in range(1, len(A)):
                counter = get_counter(A[i])
                for i, (c1, c2) in enumerate(zip(min_counter, counter)):
                  min_counter[i] = min(c1, c2)

            res = []
            for i, cnt in enumerate(min_counter):
                c = chr(ord_a + i)
                res += [c] * cnt

          return res
          ```
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
                    """
                    deduplicate
                    """
                    d1[n] = False

            return intersection
          ```
    * 350: Intersection of Two Arrays II (E)
      * Description:
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
  * **Number**:
    * 041: First missing positive (H)
      * Description:
        * Given an unsorted integer array, **find the smallest missing positive integer**.
        * example:
          * Input: [1,2,0] Output: 3
          * Input: [3,4,-1,1], Output: 2
      * Ref
        * https://leetcode.com/problems/first-missing-positive/discuss/17073/Share-my-O(n)-time-O(1)-space-solution
        * https://leetcode.com/problems/first-missing-positive/discuss/17071/My-short-c%2B%2B-solution-O(1)-space-and-O(n)-time
      * The idea is **like you put k balls into k+1 bins**, there must be a bin empty, the empty bin can be viewed as the missing number.
      * For example:
        * if length of n is 3:
          * The ideal case would be: [1, 2, 3], then the first missing is 3+1 = 4
          * The missing case may be: [1, None, 3]  5, then the first missing is 2
      * Approach1: Sorting, Time:O(nlogn), Space:O(logn~n)
      * Approach2: Generate a sorted array-1, Time O(n), Space O(n)
        * Use extra space to keep the sorted positve numbers.
        * Python
          ```python
          def firstMissingPositive(self, nums: List[int]) -> int:
            n = len(nums)

            sorted_positive = [None] * n

            for num in nums:
                if 1 <= num <= n:
                    sorted_positive[num-1] = num

            first_missing = n + 1
            for idx, num in enumerate(sorted_positive):
                if num is None:
                    first_missing = idx + 1
                    break

            return first_missing
          ```
      * Approach3: Generate a sorted array-2, Time O(n), Space O(1)
         * Each number will be put in its right place at most once after first loop *
         * Traverse the array to find the unmatch number
         * But this is not a good idea since we change the input array.
         * Python
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
    * 287: Find the Duplicate Number (M)
      * Description:
        * Given an array nums **containing n + 1 integers** where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. **Assume that there is only one duplicate number, find the duplicate one.**
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
                if slow is fast:
                    break

            target = nums[0]
            while True:
                # the intersection may be the first element
                if target is slow:
                    break
                target = nums[target]
                slow = nums[slow]

            return target
          ```
    * 765: Couples Holding Hands (H)
  * Other:
    * 277: Find the **Celebrity** (M)
      * Ref:
        * https://pandaforme.github.io/2016/12/09/Celebrity-Problem/
      * Description:
        * The definition of a celebrity is that all the other n - 1 people know him/her but he/she does not know any of them.
      * Approach1:
        1. Find the **celebrity candidate**
        2. Check if the candidate is the celebrity
           * Check the people before the celebrity candidate:
              * The celebrity does not know them but they know the celebrity.
           * Check the people after the celebrity candidate:
             * They should know the celebrity
         * Python
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
    * 134: Gas Station (M)
      * Description:
        * There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
        * You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.
        * Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.
      * Approach1: Time:O(n), Space:O(1)
        * Concept:
          * rule1:
            * **If car starts at A and can not reach B. Any station between A and B can not reach B.**
            * If A can't reach B, and there exists C between A & B which can reach B, then A can reach C first, then reach B from C, which is conflict with our init statement: A can't reach B. so, the assume that such C exists is invalid.
          * rule2
            * **If the total number of gas is bigger than the total number of cost, there must be a solution.**
            * [Proof](https://leetcode.com/problems/gas-station/discuss/287303/Proof%3A-if-the-sum-of-gas-greater-sum-of-cost-there-will-always-be-a-solution)
        * Python
          ```python
          def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
            not_found = -1
            start = 0
            total_tank = cur_tank = 0

            for i, (g, c) in enumerate(zip(gas, cost)):
                remain = g - c
                cur_tank += remain
                total_tank += remain

                if cur_tank < 0:
                    start = i + 1
                    cur_tank = 0


            return start if total_tank >= 0 else not_found
           ```
    * 128: Longest Consecutive Sequence (H)
      * Description:
        * Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
        * Your algorithm should run in O(n) complexity.
      * Approach1: Sorting + DP, Time:O(nlong), Space:O(n)
      * Approach2: Hash Set, Time:O(n), Space:O(n)
    * 845: Longest Mountain in Array
### Matrix
 * Transformation
   * 289: Game of Life (M)
      * Description:
        * Rules:
          * Any live cell with fewer than two live neighbors dies, as if caused by under-population.
          * Any live cell with two or three live neighbors lives on to the next generation.
          * Any live cell with more than three live neighbors dies, as if by over-population..
          * Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
        * At 8 neihgbors for each cells
      * Approach1: Time: O(mn), Space: O(mn)
        * Python
          ```python
          class Status(object):
              DEAD = 0
              LIVE = 1


          DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, -1),
                        (-1, 0), (-1, 1), (0, 1), (1, 1))


          def gameOfLife(self, board: List[List[int]]) -> None:
              def neighbors(r, c):
                  for d in DIRECTIONS:
                      nr, nc = r + d[0], c + d[1]
                      if 0 <= nr < row and 0 <= nc < col:
                          yield nr, nc

              row, col = len(board), len(board[0])
              c_board = [[board[r][c] for c in range(col)] for r in range(row)]

              for r in range(row):
                  for c in range(col):
                      live_neighbors = 0
                      for nr, nc in neighbors(r, c):
                          #print(nr, nc)
                          if c_board[nr][nc] == Status.LIVE:
                              live_neighbors += 1

                      if c_board[r][c] == Status.LIVE:
                          if live_neighbors < 2 or live_neighbors > 3:
                              board[r][c] = Status.DEAD

                      else:
                          if live_neighbors == 3:
                              board[r][c] = Status.LIVE
          ```
      * Approach2: Time: O(mn), Space: O(1)
        * Use two temp status, live_2_dead and dead_2_live
        * Python
          ```python
          class Status(object):
            DEAD_2_LIVE = -1
            DEAD = 0
            LIVE = 1
            LIVE_2_DEAD = 2

          DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, -1),
                        (-1, 0), (-1, 1), (0, 1), (1, 1))

          def gameOfLife(self, board: List[List[int]]) -> None:
              def neighbors(r, c):
                  for d in DIRECTIONS:
                      nr, nc = r + d[0], c + d[1]
                      if 0 <= nr < row and 0 <= nc < col:
                          yield nr, nc

              row, col = len(board), len(board[0])

              for r in range(row):
                  for c in range(col):
                      live_neighbors = 0
                      for nr, nc in neighbors(r, c):

                          """
                          Status.LIVE and Status.LIVE_2_DEAD
                          """
                          if board[nr][nc] >= Status.LIVE:
                              live_neighbors += 1

                      if board[r][c] == Status.LIVE:
                          if live_neighbors < 2 or live_neighbors > 3:
                              board[r][c] = Status.LIVE_2_DEAD

                      else:
                          if live_neighbors == 3:
                              board[r][c] = Status.DEAD_2_LIVE

              for r in range(row):
                  for c in range(col):
                      if board[r][c] == Status.LIVE_2_DEAD:
                          board[r][c] = Status.DEAD
                      elif board[r][c] == Status.DEAD_2_LIVE:
                          board[r][c] = Status.LIVE
          ```
      * follow up: Infinite array
        * [Solution](https://leetcode.com/problems/game-of-life/discuss/73217/Infinite-board-solution/201780)
   * 723：Candy Crush (M)
     * Approach1: Time: O(mn)^2, Space:O(1)
       * Time
         * In the worst case, we might crush only 3 candies each time.
         * so tottal mn/3 iteration, each iteration costs 3mn
       * Python
        ```python
        def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
          if not board or not board[0]:
              return []

          """
          for crush_size k, consider to use sliding window and memo
          """
          EMPTY = 0
          crush_size = 3
          move_step = crush_size - 1
          row, col = len(board), len(board[0])

          while True:
              need_crush = False

              # label the horizontal crush
              for r in range(row):
                  for c in range(col-move_step):
                      if board[r][c] != EMPTY and \
                          abs(board[r][c]) == abs(board[r][c+1]) == abs(board[r][c+2]):
                          board[r][c] = board[r][c+1] = board[r][c+2] = -abs(board[r][c])
                          need_crush = True


              # label the vertical crush
              for c in range(col):
                  for r in range(row-move_step):
                      if board[r][c] != EMPTY and \
                          abs(board[r][c]) == abs(board[r+1][c]) == abs(board[r+2][c]):
                          board[r][c] = board[r+1][c] = board[r+2][c] = -abs(board[r][c])
                          need_crush = True

              if not need_crush:
                  break

              # crush and gravity
              for c in range(col):
                  border = row-1
                  """
                  scan the col and move non-zero val to the border
                  """
                  for r in range(row-1, -1, -1):
                      if board[r][c] > 0:
                          board[border][c] = abs(board[r][c])
                          border -= 1

                  for r in range(border, -1, -1):
                      board[r][c] = EMPTY

          return board
        ```
 * **Rotate**:
   * 048: Rotate Image
     * Approach1: Transpose and Reverse, Time:O(n^2), Space:O(1)
       * Python
        ```python
        def rotate(self, matrix: List[List[int]]) -> None:
          n = len(matrix)
          """
          Transpose, note that c starting from r
          """
          for r in range(n):
              for c in range(r, n):
                  matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]

          # reverse each row
          for r in range(n):
              matrix[r].reverse()
        ```
     * Approach2: Layer by Layer, Time:O(n^2), Space:O(1)
       * Python
         * Implementation1:
           * Python
            ```python
            def rotate(self, matrix: List[List[int]]) -> None:
              n = len(matrix)
              layers = n // 2

              for layer in range(layers):
                  start = layer
                  end = n-1-layer
                  """
                  if n = 4, layer = 2
                  top left:   [0][0], [0][0]
                  down right: [3][3], [2][2]
                  """
                  for offset in range(end-start):
                      tmp = matrix[start][start+offset]
                      matrix[start][start+offset] = matrix[end-offset][start]
                      matrix[end-offset][start] = matrix[end][end-offset]
                      matrix[end][end-offset] = matrix[start+offset][end]
                      matrix[start+offset][end] = tmp
            ```
         * Implementation2:
           * Python
           *
   * Approach1: m*n arryay, Time:O(mn), Space:O(mn)
     * Python
      ```python
      def rotate_v2(a):
        """
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
   * Approach2: n-n array, Time:O(n^2), Space:O(1)
     * See 048
 * **Spiral**
   * 054:	Spiral Matrix (M)
     * Description:
       * Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
     * Approach1: Simulation, Time:O(mn), Space:O(mn)
       * Draw the path that the spiral makes. We know that the path should turn clockwise whenever it would go out of bounds or into a cell that was previously visited.
       * Python
        ```python
        def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          if not matrix or not matrix[0]:
              return []

          row, col = len(matrix), len(matrix[0])
          visited = [[False for _ in range(col)] for _ in range(row)]

          """
          4 directions top, right, bottom, left
          """
          DIRECTIONS = ((0, 1), (1, 0), (0,-1), (-1, 0))
          spiral_order = []
          r = c = d = 0
          for _ in range(row*col):
              spiral_order.append(matrix[r][c])
              visited[r][c] = True

              cr, cc = r + DIRECTIONS[d][0], c + DIRECTIONS[d][1]

              if 0 <= cr < row and 0 <= cc < col and not visited[cr][cc]:
                  r, c = cr, cc
              else:
                  # change direction
                  d = (d +1) % 4
                  r, c = r + DIRECTIONS[d][0], c + DIRECTIONS[d][1]

          return spiral_order
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
        def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          def spiral_traverse(r1, c1, r2, c2):
              # Top, go right, c1 to c2
              for c in range(c1, c2+1):
                  yield r1, c

              # Right, go down , r1+1 to r2
              for r in range(r1+1, r2+1):
                  yield r, c2

              """
              prevent duplicated traverse for single row and single col cases
              """
              if r1 == r2 or c1 == c2:
                return

              # Down, c2-1 to c1
              for c in range(c2-1, c1-1, -1):
                  yield r2, c

              # Left, go up  r2-1 to r1-1
              for r in range(r2-1, r1, -1):
                  yield r, c1

          if not matrix:
              return []

          row, col = len(matrix), len(matrix[0])

          """
          top, right, bottom, left
          """
          r1 = c1 = 0
          r2, c2 = row-1, col-1

          spiral_order = []
          while r1 <= r2 and c1 <= c2:
              for r, c in spiral_traverse(r1, c1, r2, c2):
                  spiral_order.append(matrix[r][c])

              # move to next layer
              r1, c1 = r1 +1, c1 + 1
              r2, c2 = r2 - 1, c2 - 1

          return spiral_order
        ```
   * 059:	Spiral Matrix II (M)
     * Description:
       * Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
     * Approach1:S imulation, Time:O(mn), Space:O(1)
       * Do not need visited matrix now
       * Python
        ```python
        def generateMatrix(self, n: int) -> List[List[int]]:
          row = col = n
          sparial_m = [[None for _ in range(col)] for _ in range(row)]

          """
          top, right, bottom, left
          """
          DIRECTIONS = ((0, 1), (1, 0), (0,-1), (-1, 0))
          r = c = d = 0
          for val in range(1, row*col+1):
              sparial_m[r][c] = val

              cr, cc = r + DIRECTIONS[d][0], c + DIRECTIONS[d][1]
              if 0 <= cr < row and 0 <= cc < col and sparial_m[cr][cc] is None:
                  r, c = cr, cc
              else:
                  d = (d +1) % 4
                  r, c = r + DIRECTIONS[d][0], c + DIRECTIONS[d][1]

          return sparial_m
        ```
     * Approach2: Layer by Layer, Time:O(mn), Space:O(1)
       * Python
        ```python
        def generateMatrix(self, n: int) -> List[List[int]]:
          def spiral_traverse(r1, c1, r2, c2):
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
              for r, c in spiral_traverse(r1, c1, r2, c2):
                  matrix[r][c] = val
                  val += 1

              r1, c1 = r1 + 1, c1 + 1
              r2, c2 = r2 - 1, c2 - 1

          return matrix
        ```
 * **Search**:
   * 074:	Search a 2D Matrix (M)
     * Description:
       * Integers in each row are sorted from left to right.
       * The first integer of each row is greater than the last integer of the previous row.
     * Approach1: Brute Force, Time:O(mn), Space:O(1)
     * Approach2: Search Space Reduction, O(m+n), Space:O(1)
       * Starting from bottom left of the matrix (or from top right ?)
         * if the target is greater than the cur val, go right
         * if the target is smaller than the cur val, go up
       * Python
          ```python
          def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            if not matrix or not matrix[0] or target < matrix[0][0] or target > matrix[-1][-1]:
                return False

            row, col = len(matrix), len(matrix[0])
            is_found = False

            r, c = row-1, 0
            while r >= 0 and c < col:
                if target == matrix[r][c]:
                    is_found = True
                    break

                # truncate the col
                elif target > matrix[r][c]:
                    c += 1
                # truncate the row
                else:
                    r -= 1

            return is_found
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
            if not matrix or not matrix[0] or \
                target < matrix[0][0] or target > matrix[-1][-1]:
                return False

            row, col = len(matrix), len(matrix[0])
            left, right = 0, row * col - 1
            is_found = False

            while left <= right:
                mid = (left + right) // 2
                r, c = mid // col, mid % col

                if target == matrix[r][c]:
                    is_found = True
                    break

                elif target > matrix[r][c]:
                    left = mid + 1

                else:
                    right = mid - 1

            return is_found
         ```
   * 240:	Search a 2D Matrix II (M)
     * Description
       * Integers in **each row** are sorted in **ascending from left to right**.
       * Integers in **each column** are sorted in **ascending** from top to bottom.
       * example:
          ```txt
          [
          [1,   4,  7, 11, 15],
          [2,   5,  8, 12, 19],
          [3,   6,  9, 16, 22],
          [10, 13, 14, 17, 24],
          [18, 21, 23, 26, 30]
          ]
          ```
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

          if not matrix or not matrix[0] or target < matrix[0][0] or target > matrix[-1][-1]:
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
     * Approach3: **Search Space Reduction**, Time:O(m + n), Space:O(1)
       * Starting from bottom left of the matrix (or from top right ?)
       * if the target is greater than the cur val, go right
       * if the target is smaller than the cur val, go up
       * Python
        ```python
        def searchMatrix(self, matrix, target):
          if not matrix or not matrix[0] or target < matrix[0][0] or target > matrix[-1][-1]:

          row, col = len(matrix), len(matrix[0])
          is_found = False

          r, c = row - 1, 0
          while r >= 0 and c < col:
              if target == matrix[r][c]:
                  is_found = True
                  break
              # truncate the col
              elif target > matrix[r][c]:
                  c += 1
              # truncate the row
              else:
                  r -= 1

          return is_found
        ```
     * Approach4: **Divide and Conquer**, Time:O(nlog(m/n)), Space:O(log(n))
       * Time complexity:
         * n is rows, m is column
         * Recursvie relation: T(m, n) = 2T(m/2, n/2) + log(m)
         * complexity: O(nlog(m/n))
         * https://stackoverflow.com/questions/2457792/how-do-i-search-for-a-number-in-a-2d-array-sorted-left-to-right-and-top-to-botto/2458113#2458113
       * Divide the matrix into four parts.
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
            def bin_search_col(col_num, top, bottom):
                is_found = False

                while top <= bottom:
                    mid = (top + bottom) // 2

                    if target == matrix[mid][col_num]:
                        is_found = True
                        break
                    elif target > matrix[mid][col_num]:
                        top = mid + 1
                    else:
                        bottom = mid -1

                return is_found, top


            if not matrix or not matrix[0] \
                or target < matrix[0][0] or target > matrix[-1][-1]:
                return False

            row, col = len(matrix), len(matrix[0])

            # top-left corner
            r1 = c1 = 0
            # bottom-right corner
            r2, c2 = row-1, col-1

            stack = [(r1, c1, r2, c2)]
            is_found = False
            while stack:
                r1, c1, r2, c2 = stack.pop()

                # check boundary
                if r1 > r2 or c1 > c2 \
                    or target < matrix[r1][c1] or target > matrix[r2][c2]:
                    continue

                mid_c = (c1 + c2) // 2
                is_found, mid_r = bin_search_col(mid_c, r1, r2)

                if is_found:
                    break

                # bottom-left block
                stack.append((mid_r, c1, r2, mid_c-1))

                # top-right block
                stack.append((r1, mid_c+1, mid_r-1, c2))

            return is_found
         ```
 * 378:	**Kth Smallest Element** in a Sorted Matrix (M)
   * Description
     * Given a n x n matrix **where each of the rows and columns are sorted in ascending order**, find the kth smallest element in the matrix.
   * Approach1: Brute Force, min-heap Time:O(nm(logk)), Space:O(k)
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

      def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
          if not matrix or not matrix[0]:
              return None

          row, col = len(matrix), len(matrix[0])

          min_heap = []
          for i in range(min(row, k)):
              min_heap.append(Element(r=i, c=0, val=matrix[i][0]))
          heapq.heapify(min_heap)

          kth_min = None
          cnt = 0
          while min_heap and cnt < k:
              r, c, kth_min = min_heap[0].r, min_heap[0].c, min_heap[0].val
              cnt += 1
              if c + 1 < col:
                heapq.heapreplace(min_heap, Element(r, c+1, matrix[r][c+1]))
              else:
                heapq.heappop(min_heap)

          return kth_min if cnt == k else None
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

          if not matrix or not matrix[0]:
              return

          row, col = len(matrix), len(matrix[0])
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
          # starting from 1
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
 * 766: Toeplitz Matrix (E)
   * Approach1: row by row
     * Python
      ```python
      def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        if not matrix or not matrix[0]:
            return True

        row, col = len(matrix), len(matrix[0])
        is_toeplitz = True
        for r in range(1, row):
            for c in range(1, col):
                if matrix[r][c] != matrix[r-1][c-1]:
                    is_toeplitz = False
                    break
            else:
                continue
            break

        return is_toeplitz
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
        * left: The values before left boundary are less than the target.
        * right: the values after right boundary are greater than the target.
      * If the target not found:
        * return not_found (-1)
      * Python
        ```python
        def guessNumber(self, n):
          res = None
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
    * Description:
      * Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
      * example:
        * Input: [1,3,5,6], 2
        * Output:[1]
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
          def bin_search_right(arr, left, right):
              target_idx = None
              while left <= right:
                  mid = (left + right) // 2
                  if target == arr[mid]:
                      target_idx = mid
                      break
                  elif target < arr[mid]:
                      right = mid -1
                  else :
                      left = mid + 1
              else:
                  target_idx = left

              return target_idx

          if not nums:
              return 0

          return bin_search_right(nums, 0, len(nums)-1)
        ```
  * 658: Find K Closest Elements in a sorted array (M)
    * Approach1: Expand from center, Time:O(logn + k), Space:O(1)
      * Python
        ```python
        def findClosestElements(self, arr, k, x):
          def bin_search_right(left, right, target):
              target_idx = None
              while left <= right:
                  mid = (left + right) // 2
                  if target == arr[mid]:
                      target_idx = mid
                      break
                  elif target < arr[mid]:
                      right = mid -1
                  else :
                      left = mid + 1
              else:
                  target_idx = left

              return target_idx

          if not arr or x is None or k <= 0:
              return []

          if x <= arr[0]:
              return arr[:k]

          if x >= arr[-1]:
              return arr[-k:]

          # center
          target_idx = bin_search_right(0, len(arr)-1, x)
          if target_idx > 0 and x - arr[target_idx-1] < arr[target_idx] -x:
              target_idx -= 1

          # expand from center
          l, r = target_idx-1, target_idx+1
          while (r - l - 1) < k and l >= 0 and r < len(arr):
              if (arr[r] - x) < (x - arr[l]):
                  r += 1
              else:
                  l -= 1

          while (r - l - 1) < k and l >= 0:
              l -= 1

          while (r - l - 1) < k and r < len(arr):
              r += 1

          # l + 1 to r - 1
          return arr[l+1: r]
        ```
    * Approach2: One binsearch, Time:O(log(n-k)), Space:O(1)
      * Ref:
        * https://leetcode.com/problems/find-k-closest-elements/discuss/106426/JavaC%2B%2BPython-Binary-Search-O(log(N-K)-%2B-K)
      * FAQ:
        * Why the mid can represent the whole array ?
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
        if isBadVersion(1) or n == 1:
            return 1

        l, r = 2, n
        while l <= r:
            mid = (l + r) // 2
            if isBadVersion(mid):
                r = mid - 1
            else:
                l = mid + 1

        return l
        ```
  * 162: Find **Peak Element** (M)
    * Description:
      * Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.
      * **The array may contain multiple peaks,** in that case return the index to any one of the peaks is fine.
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
          if not nums:
              return None

          peak = None
          for i in range(0, len(nums)-1):
              if nums[i] > nums[i+1]:
                  peak = i
                  break
          else:
              peak = len(nums)-1

          return peak
        ```
    * Approach2: Binary Search, Time:O(logn), Space:O(1)
      * Peak is not unique, we can return any peak in the nums
      * Python
        ```python
        def findPeakElement(self, nums):

          if not nums:
              return None

          n = len(nums)
          peak = None
          l, r = 0, n-1
          while l <= r:

              mid = (l + r) // 2
              if mid == n-1:
                  peak = mid
                  break

              # descending order, go left
              if nums[mid] > nums[mid+1]:
                  r = mid - 1

              # ascending order, go right
              else:
                  l = mid + 1
          else:
              peak = l

          return peak
        ```
  * 034: Find **First and Last Position of Element** in Sorted Array (M)
    * Description:
      * Given an array of integers nums sorted in ascending order, find the **starting** and **ending** position of a given target value.
      * If the target is not found in the array, return [-1, -1].
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
          def binary_search_left(left, right, target):
              l, r = left, right
              while l <= r:
                  mid = (l + r) // 2
                  if target <= nums[mid]:
                      r = mid - 1
                  else:
                      l = mid + 1

              res = not_found
              if l <= right and nums[l] == target:
                  res = l
              return res

          def binary_search_right(left, right, target):
              l, r = left, right
              while l <= r:
                  mid = (l + r) // 2
                  if target >= nums[mid]:
                      l = mid + 1
                  else:
                      r = mid - 1

              res = not_found
              if r >= left and nums[r] == target:
                  res = r
              return res


          n = len(nums)
          not_found = -1
          if n == 0:
              return [not_found, not_found]

          s = binary_search_left(0, n-1, target)
          if s == not_found:
              e = not_found
          else:
              e = binary_search_right(s, n-1, target)

          return [s, e]
        ```
  * 1150: Check If a Number Is Majority Element in a Sorted Array (E)
    * Same concept as 034
    * Description
      * Example:
          ```txt
          [ x x x x x x x . . . . . . ]   # majority at the beginning
          [ . . . x x x x x x x . . . ]   # majority at the middle
          [ . . . . . . x x x x x x x ]   # majority at the ending
          ```
    * Approach1: Linear Search, Time:O(n)
    * Approach2: Binary Search, Time:O(logn)
      * Python
        ```python
        def isMajorityElement(self, nums: List[int], target: int) -> bool:
          def binary_search_left(left, right, target):
              l, r = left, right
              while l <= r:
                  mid = (l + r) // 2
                  if target <= nums[mid]:
                      r = mid - 1
                  else:
                      l = mid + 1

              res = not_found
              if l <= right and nums[l] == target:
                  res = l
              return res

          def binary_search_right(left, right, target):
              l, r = left, right
              while l <= r:
                  mid = (l + r) // 2
                  if target >= nums[mid]:
                      l = mid + 1
                  else:
                      r = mid - 1

              res = not_found
              if left <= r and nums[r] == target:
                  res = r
              return res


          not_found = -1
          n = len(nums)

          m_left = binary_search_left(0, n-1, target)
          if m_left == not_found:
              return False
          m_right = binary_search_right(m_left, n-1, target)

          return (m_right - m_left + 1) > (n // 2)
        ```
  * 275: H-Index II (M)
    * Description:
      * Given an array of citations sorted in ascending order (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.
      *  According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."
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
              hidx = max_hidx - idx
              if cita >= hidx:
                  res = hidx
                  break

          return res
        ```
    * Approach2: Binary Search: O(log(n))
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
          idx = 0
          max_hidx = len(citations)

          l, r = 0, len(citations)-1

          while l <= r:
              mid = (l + r) // 2
              hidx = max_hidx - mid

              if citations[mid] == hidx:
                  idx = mid
                  break

              # find the smaller hidx
              elif citations[mid] < hidx:
                  l = mid + 1
              # find the bigger hidx
              else:
                  r = mid - 1

          else:
              idx = l

          return max_hidx - idx
        ```
  * 033 & 081: **Search** in **Rotated Sorted Array** (M)
    * 033: unique
    * 081: allow duplicates
    * For duplicates, the key is how to handle the case like:
      ```txt
         nums  [4, 4, 1, 2, 3, 4, 4 ]
         idx    0  1  2  3  4  5  6
                l  p           p* r
      ```
      * In such case, if we choose a pivot with value 4, we can not know the pivot is in the left part (p) or right part (p*)
        * in the left part, nums[l:p] is ascending order
        * in the right part, nums[p*:r] is ascending order
      * One solution is to move l to right until nums[left] != nums[right]
    * Approach1: Linear Search, Time:O(n), Space:O(1)
    * Approach2: Get Rotated Index (unique value), Time:O(logn), Space:O(1)
      * See 153 and 154
    * Approach3: One Pass binary Search (unique value), Time:O(logn), Space:O(1)
      * The key idea is to find the non-roated array
        * example:
          * case1: left  pivot rotate_idx right
            * left to pivot is un-rotated part
          * case2: left rotate_idx  pivot  right
            * pivot to right is un-rotated part
      * Python
        ```python
        def search(self, nums: List[int], target: int) -> int:
          def search_in_rotated_array(left, right):
              start = left
              l, r = left , right
              res = NOT_FOUND
              while l <= r:
                  mid = (l + r) // 2
                  if target == nums[mid]:
                      res = mid
                      break

                  elif nums[mid] >= nums[start]:
                      if nums[start] <= target < nums[mid]:
                          r = mid -1
                      else:
                          l = mid + 1

                  else:
                      if nums[mid] < target <= nums[r]:
                          l = mid + 1
                      else:
                          r = mid - 1
              return res

          NOT_FOUND = -1
          if not nums:
              return NOT_FOUND
          return search_in_rotated_array(0, len(nums)-1)
        ```
    * Approach4: One Pass binary Search (duplicated value), Time:O(logn)~O(n), Space:O(1)
      * Note, don't forget to update start
      * Python
        ```python
        def search(self, nums: List[int], target: int) -> bool:
          def search_in_rotated_array(left, right):
              start = left
              l, r = left , right
              res = False
              while l <= r:
                  mid = (l + r) // 2
                  if target == nums[mid]:
                      res = True
                      break

                  elif nums[mid] >= nums[start]:
                      if nums[start] <= target < nums[mid]:
                          r = mid -1
                      else:
                          l = mid + 1

                  else:
                      if nums[mid] < target <= nums[r]:
                          l = mid + 1
                      else:
                          r = mid - 1
              return res

          if not nums:
              return False

          """
          Previous algorithm can deal with duplicated items
          if the head of array is not a duplicated item
          special case: [4, 4, 1, 2, 3, 4, 4]
          """
          left, right = 0, len(nums)-1
          while left < right and nums[left] == nums[right]:
              left += 1
          return search_in_rotated_array(left, right)
        ```
  * 153 Find **Minimum** in **Rotated Sorted Array** (M)
    * unique val in the array
    * Approach1: Linear Search, Time:O(n), Space:O(1)
    * Approach2: Find Rotated index (unique value), Time:O(logn), Space:O(1)
      * Python
        ```python
        def findMin(self, nums: List[int]) -> int:
          def find_rotated_idx(left, right):
            start, end = left, right
            # the array is not rotated
            if nums[end] >= nums[start]:
                return start

            rotated_idx = NOT_FOUND
            l, r = left ,right
            while l <= r:

                mid = (l+r) // 2
                if nums[mid] > nums[mid+1]:
                    rotated_idx = mid + 1
                    break
                # mid is in the left part, search right
                if nums[mid] >= nums[start]:
                    l = mid + 1
                # mid is in the right part, search left
                else:
                    r = mid - 1

            return rotated_idx

          NOT_FOUND = -1
          if not nums:
              return NOT_FOUND
          rotated_idx = find_rotated_idx(left=0, right=len(nums)-1)
          return nums[rotated_idx]
        ```
  * 154: Find **Minimum** in **Rotated Sorted Array** II (H)
    * allow duplicated val in the array
    * Approach1: Find Rotated index (duplicated value), Time:O(logn~n), Space:O(1)
      * Python
        ```python
        def findMin(self, nums: List[int]) -> int:
          def find_rotated_idx(left, right):
            start, end = left, right
            # the array is not rotated
            if nums[end] >= nums[start]:
                return start

            rotated_idx = NOT_FOUND
            l, r = left ,right
            while l <= r:

                mid = (l+r) // 2
                if nums[mid] > nums[mid+1]:
                    rotated_idx = mid + 1
                    break
                # mid is in the left part, search right
                if nums[mid] >= nums[start]:
                    l = mid + 1
                # mid is in the right part, search left
                else:
                    r = mid - 1

            return rotated_idx

          NOT_FOUND = -1
          if not nums:
              return NOT_FOUND

          left, right = 0, len(nums)-1
          """
          [2,2,2,2,4,5,6,7,0,1,2,2,2,2]
          """
          while left < right and nums[left] == nums[right]:
              left += 1

          rotated_idx = find_rotated_idx(left, right)

          return nums[rotated_idx]
        ```
  * 222: Count **Complete Tree Nodes** (M)
    * Description:
      * Given a complete binary tree, count the number of nodes.
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

          def get_depth(node):
              depth = 0

              while node and node.left:
                  node = node.left
                  depth += 1

              return depth

          def check_leaf_exist(node_idx, root, d):
              left, right = 0, 2**d-1
              cur = root
              for _ in range(d):
                  mid = (left + right) // 2
                  if node_idx <= mid:
                      right = mid
                      cur = cur.left
                  else:
                      left = mid + 1
                      cur = cur.right

              return cur is not None

          if not root:
              return 0

          d = get_depth(root)
          left, right = 1, 2**d-1

          while left <= right:

              mid = (left + right) // 2

              if check_leaf_exist(mid, root, d):
                  left = mid + 1
              else:
                  right = mid - 1

            """
            The tree contains 2**d - 1 nodes on the first (d - 1) levels
            And leaf nodes in the last level
            """
          return (2**d-1) + left
          ```
  * 004: **Median** of **Two Sorted Arrays** (H)
    * Description:
      * There are two sorted arrays nums1 and nums2 of size m and n respectively.
      * Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
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
      * Python:
        ```python
        MAX_VAL = float('inf')
        MIN_VAL = float('-inf')
        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            n1, n2 = len(nums1), len(nums2)

            if n1 == n2 == 0:
                return 0

            if n1 > n2:
                n1, n2 = n2, n1
                nums1, nums2 = nums2, nums1

            """
            This number means how many number is smaller than the median:
            0:  0 elements are smaller than the median
            n1: all elements are smaller than the median
            total n + 1 cases starting from 0 to n
            """
            left = 0    # left boundary
            right = n1  # right boundary

            """
            The half len of merged array (merge nums1 and nums2)
            p_n1 + p_n2 = half_len
            """
            half_len = (n1 + n2 + 1) // 2
            median = None

            while left <= right:

                p_n1 = (left + right) // 2
                p_n2 = half_len - p_n1
                """
                p_n1 == 0 means nothing is there on left side. Use -INF as sentinel
                p_n1 == n1 means there is nothing on right side. Use +INF as sentinel
                for example, if nums1 = [1, 2, 3]
                case0: [-INF], [1, 2, 3]  # p_n1 == 0, [nums1-left][nums2][nums1-right] =
                case1: [1], [2,3]         # p_n1 == 1
                case2: [1, 2], [3]        # p_n1 == 2
                case3: [1, 2, 3], [+INF]  # p_n1 == 3 == n1, [nums1-left][nums2][nums1-right]
                """
                max_left_n1 = MIN_VAL if p_n1 == 0 else nums1[p_n1-1]
                min_right_n1 = MAX_VAL if p_n1 == n1 else nums1[p_n1]

                max_left_n2 = MIN_VAL if p_n2 == 0 else nums2[p_n2-1]
                min_right_n2 = MAX_VAL if p_n2 == n2 else nums2[p_n2]

                if max_left_n1 <= min_right_n2 and max_left_n2 <= min_right_n1:
                    # even
                    if (n1 + n2) % 2 == 0:
                        median = (max(max_left_n1, max_left_n2) +
                                  min(min_right_n1, min_right_n2)) / 2
                    else:
                        """
                        p_n1 + p_n2 = half_len
                        """
                        median = float(max(max_left_n1, max_left_n2))
                    break

                elif max_left_n1 > min_right_n2:
                    """
                    shrink n1 left part
                    """
                    right = p_n1 - 1
                else:
                    """
                    max_left_n2 > min_right_n1
                    shrink n2 left part, increase n1 left part
                    """
                    left = p_n1 + 1

            return median
            ```
  * 300: Longest Increasing Subsequence (**LIS**) (M)
    * Approach1: Bottom up DP, Time:O(n^2), Space:O(n)
      * memo[i] is the longest increasing subsequece of nums[:i+1]
        * possible prev indices are from 0 to i
        * **i should be included**
      * Python
        ```python
        def lengthOfLIS(self, nums: List[int]) -> int:
          if not nums:
              return 0

          memo = [1] * len(nums)
          max_lcis = 1
          for i in range(len(nums)):
              lcis = 1
              for j in range(i):
                  if nums[i] > nums[j]:
                      lcis = max(lcis, 1 + memo[j])

              memo[i] = lcis
              max_lcis = max(max_lcis, lcis)

          """
          max lis is not memo[-1]
          """
          return max_lcis
        ```
    * Approach2: Binary Search, Time:O(nlogn), Space:O(n)
      * Ref:
        * https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation
        * https://www.youtube.com/watch?v=S9oUiVYEq7E
      * Concept:
        * tails is an array storing the smallest tail of all increasing subsequences with length i in tails[i-1]
         * example:
           ```txt
           len = 1:  [4], [5], [6], [3]   => tails[0] = 3
           len = 2:  [4, 5], [5, 6]       => tails[1] = 5
           len = 3:  [4, 5, 6]            => tails[2] = 6
           ```
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
          n = len(nums)

          if n <= 1:
              return n

          tails = [None] * n
          tails[0] = nums[0]
          length = 1

          for i in range(1, n):
              target = nums[i]
              l, r = 0, length-1

              while l <= r:
                  mid = (l + r) // 2

                  # do not need to update
                  if target == tails[mid]:
                      break

                  if target < tails[mid]:
                      r = mid - 1
                  else:
                      l = mid + 1
              else:
                  tails[l] = target
                  length = max(length, l+1)

          return length
        ```
  * 1060: Missing Element in Sorted Array (M)
  * 187: Repeated DNA Sequences (M)
  * 315: Count of Smaller Numbers After Self (H)
  * 354: Russian Doll Envelopes (H)
  * Other:
    * 896: Monotonic Array (E)
    * 1060: Missing Element in Sorted Array (M)
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
  * 876: Middle of the Linked List (E)
    * Approach1: Runner, Time:O(n), Space:O(1)
      * Python
        ```python
        def middleNode(self, head: ListNode) -> ListNode:
          if not head:
              return None

          slow = fast = head
          while fast and fast.next:
              fast = fast.next.next
              slow = slow.next

          return slow
        ```
  * 141: Linked List Cycle (E)
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
  * 142: Linked List Cycle II (M)
    * Similar problems:
      * 287: Find the Duplicate Number (M)
    * Given a linked list, return the node **where the cycle begins**. If there is no cycle, return null.
    * The length before the cycle beginning: **d**
    * The length of the cycle: **r**
    * Approach1: Hash Table, Time:O(n), Space:O(n)
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
     * Description
       * Remove all elements from a linked list of integers that have value val.
     * Approach1: dummy node, Time:O(n), Space:O(1)
       * Python
          ```python
          def removeElements(self, head: ListNode, val: int) -> ListNode:
            if not head:
                return head

            cur = dummy = ListNode(0)
            dummy.next = head

            while cur and cur.next:
                if cur.next.val == val:
                    cur.next = cur.next.next
                else:
                    cur = cur.next

            return dummy.next
          ```
   * 019: Remove Nth Node From End of List (M)
     * Description:
       * Given a linked list, remove the n-th node **from the end** of list and return its head.
       * Intput: 1->2->3->4->5 and n == 2
       * Output: 1->2->3->5
       * Given n will always be valid.
     * Approach1: runner + dummy node, Time:O(n), Space:O(1)
       * Python
          ```python
          def removeNthFromEnd(self, head, n):
            dummy = ListNode(0)
            fast = dummy.next = head

            for _ in range(n):
                fast = fast.next

            prev = dummy
            while fast:
                prev, fast = prev.next, fast.next

            # delete the target
            prev.next = prev.next.next

            return dummy.next
          ```
   * 083: Remove Duplicates from **Sorted** List (E)
     * Example:
       * Given a sorted linked list, delete all duplicates such that each element appear only once.
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
     * Description:
       * Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
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
   * 1171: Remove Zero Sum Consecutive Nodes from Linked List (M)
     * Description:
       * Given the head of a linked list, we repeatedly delete consecutive sequences of nodes that sum to 0 until there are no such sequences.
       * Intput: [1,2,3,-3,-2]
       * Output: [1]
     * Python, prefix sum, O(n), Space:O(n)
       * Python
        ```python
        def removeZeroSumSublists(self, head: ListNode) -> ListNode:
          if not head:
              return None

          dummy = ListNode(0)
          cur = dummy.next = head
          memo = {0: dummy}
          prefix = 0
          while cur:
              prefix += cur.val
              if prefix in memo:
                  """
                  1. remove the prefix sum from memo[prefix].next to cur-1
                  2. remove the node from memo[prefix].next to cur
                  """
                  clean = memo[prefix].next
                  p = prefix + clean.val
                  while clean is not cur:
                      memo.pop(p)
                      clean = clean.next
                      p += clean.val
                  memo[prefix].next = cur.next
              else:
                  memo[prefix] = cur

              cur = cur.next

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
    * Description:
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
            for _ in range(m-1):
                prev_end = prev_end.next

            # step2: reverse the nodes in the range
            prev, cur = prev_end, prev_end.next
            for _ in range(n-m+1):
                nxt = cur.next
                cur.next = prev
                prev, cur = cur, nxt

            # connect
            prev_end.next.next = cur
            prev_end.next = prev

            return dummy.next
        ```
  * 025: **Reverse** Nodes in **k-Group** (H)
    * Description:
      * Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.k is a positive integer and is less than or equal to the length of the linked list.
      * If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
      * example:
        * Input: 1->2->3->4->5
        * Output:
          * k = 2, 2->1->4->3->5
          * k = 3, 3->2->1->4->5
    * Approach1: Two Passes, Time:O(n), Space:O(1)
      * Python
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
        if not head:
            return None

        dummy = ListNode(0)
        dummy.next = head

        cnt = 0
        cur = head
        prev_end = dummy
        while cur:
            cnt += 1
            if cnt % k == 0:
                prev, cur = prev_end, prev_end.next
                for _ in range(k):
                    nxt = cur.next
                    cur.next = prev
                    prev, cur = cur, nxt

                next_prev_end = prev_end.next
                prev_end.next = prev
                next_prev_end.next = cur
                prev_end = next_prev_end
            else:
                cur = cur.next

        return dummy.next
      ```
  * 024: **Swap** Nodes in **Pair** (M) *
    * Description:
      * Input: 1->2->3->4
      * Output: 2->1->4->3
    * Approach1: Dummy Node, Time:O(n), Space:O(1)
      * Implementation1: 3 pointers
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
      * Implementation2: 2 pointers
        * Python
          ```python
          def swapPairs(self, head: ListNode) -> ListNode:
            if not head:
                return None

            prev = dummy = ListNode(0)
            cur = dummy.next = head

            while cur and cur.next:
                prev.next = cur.next
                cur.next = cur.next.next
                prev.next.next = cur

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
          if not head or not head.next:
              return

          # find the tail of the previous linked list
          prev_tail = head
          fast = head.next
          while fast and fast.next:
              prev_tail = prev_tail.next
              fast = fast.next.next

          # reverse the post linked list
          # ensure the 1st part has the same or one more node
          prev, cur = None, prev_tail.next
          while cur:
              nxt = cur.next
              cur.next = prev
              prev, cur = cur, nxt

          # init and connect
          prev_head = head
          prev_tail.next = None
          post_head = prev
          while post_head:
              post_nxt = post_head.next
              post_head.next = prev_head.next
              prev_head.next = post_head

              prev_head = post_head.next
              post_head = post_nxt

          return head
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
    * Description:
      * case1:
        * Input: 1->2->3->4->5, k = 2
        * Output: 4->5->1->2->3
      * case2:
        * Input: 0->1->2, k = 4
        * Output: 2->0->1
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
    * Description:
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
          def get_len(node):
              cnt = 0
              while node:
                  node = node.next
                  cnt += 1
              return cnt

          partitions = [None for _ in range(k)]

          if not root:
              return partitions

          n = get_len(root)

          base_cnt = n // k
          extra_cnt = n % k
          cur = root
          for i in range(k):
              cnt = base_cnt
              if extra_cnt:
                  cnt += 1
                  extra_cnt -=1

              partitions[i] = cur
              for _ in range(cnt-1):
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

          if not l1 and not l2:
              return None

          if not l1:
              return l2

          if not l2:
              return l1

          return merge_2_sorted_lists(l1, l2)
        ```
  * 148: Sort list (M)
      * Ref:
        * https://stackoverflow.com/questions/1525117/whats-the-fastest-algorithm-for-sorting-a-linked-list/1525419#1525419
      * Approach1: Transfer to array,  Time: O(nlog(n), Space:O(n)
      * Approach2: Top down merge sort , Time: O(nlog(n), Space:O(logn)
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

            def merge_sort(head):
                # need at least two nodes
                if not head or not head.next:
                    return head

                mid = head
                fast = head.next
                while fast and fast.next:
                    fast = fast.next.next
                    mid = mid.next

                left = head
                right = mid.next
                # split the list
                mid.next = None

                left = merge_sort(left)
                right = merge_sort(right)
                merged = merge_2_sorted_lists(left, right)
                return merged

            if not head:
                return None

            return merge_sort(head)
          ```
      * Approach3: Bottom up merge sort , Time: O(nlog(n), Space:O(1)
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
            def get_len(node):
                cnt = 0
                while node:
                    node = node.next
                    cnt += 1
                return cnt

            def split_list_with_size(cur, k):
                """
                Divide the linked list into two lists,
                while the first list contains first k ndoes
                return the second list's head
                """
                for _ in range(k-1):
                    if cur:
                        cur = cur.next

                if not cur:
                    return None

                post_head = cur.next
                cur.next = None

                return post_head

            def merge_2_sorted_lists(l1, l2, prev_end):
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

                while cur and cur.next:
                    cur = cur.next

                return cur

            if not head:
                return None

            n = get_len(head)
            prev_end = dummy = ListNode(0)
            dummy.next = head
            w_size = 1
            while w_size < n:
                prev_end = dummy
                left = dummy.next

                while left:
                    right = split_list_with_size(left, w_size)
                    next_left = split_list_with_size(right, w_size)
                    prev_end = merge_2_sorted_lists(left, right, prev_end)
                    left = next_left

                w_size *= 2

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

              __slots__ = 'node'

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
            __slots__ = ["node"]

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
            w_size = 1
            while interval < len(lists):
                left = 0
                while left + w_size < len(lists):
                    right = left+w_size
                    lists[left] = self.merge_2_sorted_Lists(lists[left], lists[left+right])
                    left += (2 * w_size)
                w_size *= 2

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

                 # append new node to the left
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

              # append new node to the left
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
      * Python
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
          while cur and carry:
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
  * Design:
    * 155: Min Stack (E)
      * Description:
        * Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
          * push(x)
            * Push element x onto stack.
          * pop()
            * Removes the element on top of the stack.
          * top()
            * Get the top element.
          * getMin()
            * Retrieve the minimum element in the stack.
      * Approach1:
        * Use extra space to keep minimum
        * Python
          ```python
          class MinStack:
            __slots__ = ['s', 'min_s']
            def __init__(self):
                """
                initialize your data structure here.
                """
                self.s = []
                self.min_s = []

            def push(self, x: int) -> None:
                self.s.append(x)

                if not self.min_s:
                    self.min_s.append(x)
                else:
                    self.min_s.append(min(self.min_s[-1], x))

            def pop(self) -> None:
                if not self.s:
                    return

                self.s.pop()
                self.min_s.pop()

            def top(self) -> int:
                return self.s[-1]

            def getMin(self) -> int:
                return self.min_s[-1]
          ```
    * 716: Max Stack (E)
      * Description:
        * Design a max stack that supports push, pop, top, peekMax and popMax.
          * push(x)
            * Push element x onto stack.
          * pop()
          *   Remove the element on top of the stack and return it.
          * top()
            * Get the element on the top.
          * peekMax()
            * Retrieve the maximum element in the stack.
          * popMax()
            * Retrieve the maximum element in the stack, and remove it. If you find more than one maximum elements, only remove the top-most one.
      * Approach1: Use 2 stacks, popMax() would be O(n)
        * Same concept like 155: Min Stack, use another stack the track the max
        * For popMax, we know what the current maximum (peekMax) is. We can pop until we find that maximum, then push the popped elements back on the stack.
        * Ref:
          * https://leetcode.com/problems/max-stack/discuss/108941/C%2B%2B-using-Two-Stack

        *
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
  * Calculator
    * 227: Basic Calculator II (M)
      * Description:
        * The expression string contains only non-negative integers, +, -, *, / operators and empty spaces . The integer division should truncate toward zero.
      * Approach1: 2 dqs, Time:O(n), Space:O(n)
        * Python
          ```python
          def calculate(self, s: str) -> int:

            num_dq = collections.deque()
            op_dq = collections.deque()
            i = 0
            while i < len(s):
                c = s[i]
                if c.isdigit():
                    n = 0
                    while i< len(s) and s[i].isdigit():
                        n = n * 10 + (ord(s[i]) - ord('0'))
                        i += 1

                    """
                    for * and /
                    """
                    if op_dq and op_dq[-1] in ['*', '/']:
                        op = op_dq.pop()
                        prev = num_dq.pop()
                        if op == '*':
                            n = prev * n
                        else:
                            n = prev // n

                    num_dq.append(n)

                elif c.isspace():
                    i += 1
                else:
                    op_dq.append(c)
                    i += 1

            res = 0
            if num_dq:
                res = num_dq.popleft()

            """
            for + and -
            """
            while num_dq and op_dq:
                op = op_dq.popleft()
                nxt = num_dq.popleft()
                if op == '+':
                    res = res + nxt
                else:
                    res = res - nxt

            return res
          ```
      * Approach2: 1 stack, Time:O(n), Space:O(n)
        * notice the negative value
          * -2 / 3 = should be case2
            * case1: -(2) // 3 = -1
            * case2: -(2 // 3 ) = 0
        * Python
          ```python
          def calculate(self, s: str) -> int:
            n_stack = []
            sign = '+'
            i = 0
            while i < len(s):
                c = s[i]
                if c.isdigit():
                    n = 0
                    while i< len(s) and s[i].isdigit():
                        n = n * 10 + (ord(s[i]) - ord('0'))
                        i += 1

                    if sign == '+':
                        n_stack.append(n)
                    elif sign == '-':
                        n_stack.append(-n)
                    elif sign == '*':
                        n_stack.append(n_stack.pop() * n)
                    else:  # '/'
                        """
                        -2/3 should be case2
                        case1: (-2) // 3 = -1
                        case2: -(2//3) = 0
                        """
                        prev = n_stack.pop()
                        if prev < 0:
                            # transfer case1 to case2
                            n_stack.append(-(-prev//n))
                        else:
                            n_stack.append(prev//n)

                elif c.isspace():
                    i += 1
                else:
                    sign = c
                    i += 1

            return sum(n_stack)
          ```
    * 224: Basic Calculator (H)
      * Description:
        * The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .
        * example:
          * " 2-1 + 2 " = 3
          *  "(1+(4+5+2)-3)+(6+8)" = 23
      * Appraoch1 2 Stacks, Time:O(n), Space:O(n)
        * Python
          ```python
          def calculate(self, s: str) -> int:
            if not s:
                return 0

            n_s = []
            sign_s = []
            result, sign = 0, 1

            i = 0
            while i < len(s):
                c = s[i]

                if c.isdigit():
                    n = 0
                    while i < len(s) and s[i].isdigit():
                        n = n * 10 + (ord(s[i]) - ord('0'))
                        i += 1
                    result += sign * n
                elif c == '+':
                    sign = 1
                    i += 1
                elif c == '-':
                    sign = -1
                    i += 1
                elif c == '(':
                    sign_s.append(sign)
                    n_s.append(result)
                    result, sign = 0, 1
                    i += 1
                elif c == ')':
                    sign = sign_s.pop()
                    prev = n_s.pop()
                    result = prev + sign * result
                    i += 1
                else:
                    i += 1

            return result
          ```
    * 772: Basic Calculator III (H)
      * Combination of 227 and 224
      * Description:
        * The expression string contains only non-negative integers, +, -, *, / operators , open ( and closing parentheses ) and empty spaces . The integer division should truncate toward zero.
        * example:
          * "1 + 1" = 2
          * " 6-4 / 2 " = 4
          * "2*(5+5*2)/3+(6/2+8)" = 21
          * "(2+6* 3+5- (3*14/7+2)*5)+3"=-12
    * 770: Basic Calculator IV (H)
  * 394: Decode String (M)
    * Description:
      * example:
        * Input:"3[a]2[bc]", Output "aaabcbc".
        * Input:"3[a2[bc]]", Output "abcbcabcbcabcbc".
    * Approach1: Use 2 stacks: Time:O(kn), Space:O(kn)
      * Description
      * Ref:
        * https://leetcode.com/problems/decode-string/discuss/87534/Simple-Java-Solution-using-Stack
      * Python
        ```python
        def decodeString(self, s: str) -> str:
          if not s:
              return ""

          buf = []
          buf_s = []
          cnt_s = []
          ord_0 = ord('0')

          i = 0
          while i < len(s):
              c = s[i]
              if c.isdigit():
                  cnt = 0
                  while i < len(s) and s[i].isdigit():
                      cnt = cnt * 10 + (ord(s[i]) - ord_0)
                      i += 1
              elif c == '[':
                  cnt_s.append(cnt)
                  buf_s.append(''.join(buf))
                  # or clean the buffer
                  buf = []
                  i += 1
              elif c == ']':
                  prefix = buf_s.pop()
                  cnt = cnt_s.pop()
                  postfix = ''.join(buf)
                  buf = [prefix + cnt*postfix]
                  i += 1
              else:
                  buf.append(c)
                  i += 1

          return ''.join(buf)
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
          low = float('-inf')
          stack = []
          is_bst = True
          for n in preorder:

              if n < low:
                  is_bst = False
                  break

              while stack and n > stack[-1]:
                  """
                  pop until find the current lower bound
                  """
                  low = stack.pop()

              stack.append(n)

          return is_bst
        ```
  * 341: Flatten Nested List Iterator (M)
    * Approach1: Use stack
      * Python
        ```python
        class NestedIterator(object):

          __slots__ = ['stack']

          def __init__(self, nestedList):
              self.stack = nestedList[::-1]

          def next(self):
              return self.stack.pop().getInteger()

          def hasNext(self):
              has_next = False
              while self.stack:
                  top = self.stack[-1]
                  if top.isInteger():
                      has_next = True
                      break
                  else:
                      self.stack.pop()
                      self.stack.extend(top.getList()[::-1])

              return has_next
        ```
    * Approach2: Use deque
      * Python
      ```python
      class NestedIterator(object):

        __slots__ = ['dq']

        def __init__(self, nestedList):
            self.dq = collections.deque(nestedList)

        def next(self):
            return self.dq.popleft().getInteger()

        def hasNext(self):
            has_next = False

            while self.dq:
                top = self.dq[0]
                if top.isInteger():
                    has_next = True
                    break
                else:
                    self.dq.popleft()
                    self.dq.extendleft(top.getList()[::-1])

            return has_next
      ```
  * 536: Construct Binary Tree from String
    * Description:
      * You need to construct a binary tree from a string consisting of parenthesis and integers.
      * "4(2(3)(1))(6(5))"
    * Approach1: one stack, Time:O(n), Space:O(N)
      * Python
        ```python
        def str2tree(self, s):
          if not s:
              return None

          root = None
          stack = []
          ORD_0 = ord('0')
          i = 0
          sign = 1
          for i in range(len(s)):
              c = s[i]
              if c == '-':
                  sign = -1
              elif c.isdigit():
                  val = ord(c)-ORD_0
                  new = TreeNode(sign*val)
                  sign = 1
                  if stack:
                      parent = stack[-1]
                      if not parent.left:
                          parent.left = new
                      else:
                          parent.right = new
                  stack.append(new)
              elif c == ')':
                  root = stack.pop()
                  print(root.val)

          return stack[0]
        ```
### Heap (Priority Queue)
  * Note:
    * For Python, heapq is min-heap
  * Merge k Sorted Lists
    * 023: Merge k Sorted Lists (H)
      * Please refer linked list section
    * 378: **Kth Smallest Element** in a Sorted Matrix (M)
      * Please refer matrix section
    * 373: **Find K Pairs** with Smallest Sums (M)
      * Approach1: Brute Force, Time:O(nmlogk), Space:O(k)
        * Complexity:
          * Space:
            * min_heap with size k
          * Time:
            * generate mn pairs and push to min_heap
        * Generate all possible pair and push to min_heap with size k
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
      * Approach2: Use Matrix to simulate merge k sorted list, Time:O(k*heap_size), Space:O(heap_size)
        * Complexity:
          * n : len(nums1), row number
          * heap_size : min(k, n)
          * Time complexity:
            * k * heap_size
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

          def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
              if not nums1 or not nums2:
                  return []

              row, col = len(nums1), len(nums2)
              min_heap = []
              for r in range(min(row, k)):
                  c = 0
                  min_heap.append(Element(r, c, nums1[r]+nums2[c]))
              heapq.heapify(min_heap)

              pairs = []
              while min_heap and len(pairs) < k:
                  r, c = min_heap[0].r, min_heap[0].c
                  if c + 1 < col:
                      heapq.heapreplace(min_heap, Element(r, c+1, nums1[r]+nums2[c+1]))
                  else:
                      heapq.heappop(min_heap)

                  pairs.append((nums1[r], nums2[c]))

              return pairs
          ```
  * **Schedule**:
    * 253: Meeting Rooms II (M)
      * Description:
        * Find the minimum requirement of the meeting rooms.
      * Approach1: Brute Force, Time: O(n^2), Space: O(1)
      * Approach2: Greedy using Min Heap: O(nlog(n)), Space: O(n)
        * Algo
          * Sort the intervals by start time
          * For every meeting room check if the minimum element of the heap is free or not.
            * If the room is free, then we extract the topmost element and add it back with the ending time of the current meeting we are processing.
            * If not, then we allocate a new room and add it to the heap.
        * We need a min_heap to track current mininum end time interval in the list
        * Python
          ```python
          def minMeetingRooms(self, intervals: List[List[int]]) -> int:
            if not intervals:
                return 0

            START, END = 0, 1
            intervals.sort(key = lambda interval: interval[START])

            min_heap = [intervals[0][END]]

            for i in range(1, len(intervals)):
                nxt = intervals[i]
                if nxt[START] >= min_heap[0]:
                    heapq.heapreplace(min_heap, nxt[END])
                else:
                    heapq.heappush(min_heap, nxt[END])

            return len(min_heap)
          ```
      * Approach3: O(m), Space:O(m)
        * If we can know the time range, this problem can be O(time_range)
    * 1094: Car Pooling (M)
      * Description:
        * You are driving a vehicle that has capacity empty seats initially available for passengers.  The vehicle only drives east.
        * Given a list of trips
        * trip[i] = [num_passengers, start_location, end_location]
      * Similar to Meeting Rooms II
      * Approach1: Brute Force, Time:O(n^2), Space:O(1)
      * Approach2: greedy, min-heap, Time:O(nlogn), Space:O(n)
        * Python
          ```python
          class Element(object):
              def __init__(self, trip):
                  self.trip = trip

              def __lt__(self, other):
                  return self.trip[END] < other.trip[END]

          def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
              if not trips:
                  return True

              trips.sort(key=lambda trip: trip[START])
              min_heap = []

              passengers = True
              for trip in trips:
                  # any passengers need to get off?
                  while min_heap and min_heap[0].trip[END] <= trip[START]:
                      pop = heapq.heappop(min_heap)
                      # release the seats
                      passengers -= pop.trip[CNT]

                  passengers += trip[CNT]
                  if passengers > capacity:
                      res = False
                      break

                  heapq.heappush(min_heap, Element(trip))

              return res
          ```
      * Approach3: diff array, Time:O(max(n, m)), Space:O(m)
        * M is the number of stations
        * Since stations cnt is limited, we can add passenger count to the start location, and remove them from the end location, and process the array the check if the current people exceeds the capacity.
        * Python
          ```python
          CNT, START, END = 0, 1, 2

          def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
              if not trips:
                  return True

              stations = [0] * 1001
              for trip in trips:
                  stations[trip[START]] += trip[CNT]
                  stations[trip[END]] -= trip[CNT]

              res = True
              passengers = 0
              for cnt in stations:
                  passengers += cnt
                  if passengers > capacity:
                      res = False
                      break

              return res
          ```
    * 1109: Corporate Flight Bookings (M)
      * Description:
        * There are n flights, and they are labeled from 1 to n.
        * We have a list of flight bookings.  The i-th booking bookings[i] = [i, j, k] means that we booked k seats from flights labeled i to j inclusive.
        * Return an array answer of length n, representing the number of seats booked on each flight in order of their label.
      * Approach1: brute force, Time:O(n^2), Space:O(n)
      * Approach2: diff array, Time:O(n), Space:O(n)
        * Python
          ```python
          def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
            START, END, CNT = 0, 1, 2
            flights = [0] * n

            for b in bookings:
                flights[b[START]-1] += b[CNT]
                # b[END]-1+1
                end = b[END]
                if end < n:
                    flights[end] -= b[CNT]

            cur_cnt = 0
            for i, cnt in enumerate(flights):
                cur_cnt += cnt
                flights[i] = cur_cnt

            return flights
          ```
    * 731: My Calendar II (M)
    * 732: My Calendar III (H)
    * 630: Course Schedule III (H)
      * Ref:
        * https://leetcode.com/problems/course-schedule-iii/discuss/104847/Python-Straightforward-with-Explanation
  * **Kth problem**:
    * Ref:
      * Summary:
        * https://leetcode.com/problems/k-closest-points-to-origin/discuss/220235/Java-Three-solutions-to-this-classical-K-th-problem.
        * heap solution: Time:O(nlogk), Space:O(k)
          * The advantage of this solution is **it can deal with real-time(online) stream data**.
        * quick select: Time:O(n~n^2), Space:O(n)
          * The disadvatage of this solution are it is n**either an online solution nor a stable one. And the K elements closest are not sorted in ascending order**.
    * 703: **Kth Largest** Element in a Stream (E)
      * Approach1: min heap, Time: O(log(k)), Space: O(k)
        * Time: O(log(k))
          * __init__: O(nlog(k))
            * n items for heap push
          * add:
            * heapreplace and heappush take O(log(k))
        * Space: O(k)
          * Capacity of heap size
        * Python:
          ```python
          class KthLargest:

            def __init__(self, k: int, nums: List[int]):
                # You may assume that nums' length ≥ k-1 and k ≥ 1.
                self.min_heap = list()
                self.cap = k
                for num in nums:
                    self._add(num)

            def _add(self, val):
                if len(self.min_heap) < self.cap:
                    heapq.heappush(self.min_heap, val)
                else:
                    # replace the minimize element in the heap
                    if val > self.min_heap[0]:
                        heapq.heapreplace(self.min_heap, val)

            def add(self, val: int) -> int:
                self._add(val)
                return self.min_heap[0]
          ```
    * 215: **Kth Largest** Element in an Array (M)
      * Note:
        * heap, quick select
      * Approach1: Sorting, Time: O(nlog(n)), Space:O(log(n)~n)
      * Approach2: min heap, Time: O(nlog(k)), Space: O(k)
        * keep the k element in the minimum heap
        * Time Complexity: O(nlog(k))
          * **heappush**
            * Append to the tail of list, and make bubble up comparison cost log(k)
          * **heapreplace**
            * Replace the min, and make bubble down operation, cost log(k)
        * Python
          ```python
          def findKthLargest(self, nums: List[int], k: int) -> int:
            min_heap = []
            for n in nums:
                if len(min_heap) < k:
                    heapq.heappush(min_heap, n)
                elif n > min_heap[0]:
                    heapq.heapreplace(min_heap, n)

            return min_heap[0]
          ```
      * Approach3: Quick select, Time: O(n~n^2), Space:O(1)
        * Python:
          ```python
          def findKthLargest(self, nums: List[int], k: int) -> int:

              def partition(start, end):
                  ran = random.randint(start, end)
                  pivot = end
                  nums[pivot], nums[ran] = nums[ran], nums[pivot]

                  border = start
                  for cur in range(start, end):
                      if nums[cur] >= nums[pivot]:
                        nums[cur], nums[border] = nums[border], nums[cur]
                        border += 1

                  nums[border], nums[pivot] = nums[pivot], nums[border]
                  return border

              def quick_select(start, end, k_largest):
                  res = None
                  while start <= end:
                      p = partition(start, end)
                      if p == k_largest:
                          res = nums[k_largest]
                          break
                      elif p > k_largest:
                          end = p - 1
                      else:
                          start = p + 1
                  return res

              if k > len(nums):
                  return None

              return quick_select(0, len(nums)-1, k-1)
          ```
      * can not use bucket sort since we don't know the range
    * 347: **Top K Frequent** Elements (M)
      * Given a non-empty array of integers, return the **k most frequent elements**.
        * Approach0: If k == 1, Hash Table, Time:O(n), Space:O(n)
        * Approach2: min-heap: Time O(nlog(k)), Space:O(n)
          * Complexity:
            * Space:
              * Counter: O(n)
              * min_heap: O(k)
            * Time:
              * O(nlogk)
          * Implementation1
            ```python
            class Element(object):

             __slots__ = ['key', 'cnt']

              def __init__(self, key, cnt):
                  self.key = key
                  self.cnt = cnt

              def __lt__(self, other):
                  return self.cnt < other.cnt

              def __gt__(self, other):
                  return self.cnt > other.cnt

            def topKFrequent(self, nums: List[int], k: int) -> List[int]:
                if not nums or k <= 0:
                  return []

                counter = collections.Counter(nums)
                min_heap = []

                for key, cnt in counter.items():
                    e = Element(key, cnt)
                    if len(min_heap) < k:
                        heapq.heappush(min_heap, e)
                    else:
                        if e > min_heap[0]:
                            heapq.heapreplace(min_heap, e)

                return [e.key for e in min_heap]
            ```
            * Implementation2, use nlargest
            ```python
            def topKFrequent(self, nums: List[int], k: int) -> List[int]:
              counter = collections.Counter(nums)
              return heapq.nlargest(k, counter.keys(), key=counter.get)
            ```
        * Approach3: quick-select, Time:O(n), Space:O(n)
          * Complexity:
            * Space: O(n)
              * Counter:O(n)
              * key_cnt array: O(n)
            * Time: O(n)
              * Quick select: O(n)
          * Python
            ```python
            class Element(object):

                __slots__ = 'key', 'cnt'

                def __init__(self, key, cnt):
                    self.key = key
                    self.cnt = cnt

                def __ge__(self, other):
                    return self.cnt >= other.cnt

            def quick_select(self, array, start, end, k_largest):
                def partition(start, end):
                    ran = random.randint(start, end)
                    pivot = end
                    array[pivot], array[ran] = array[ran], array[pivot]

                    border = start
                    for cur in range(start, end):
                        if array[cur] >= array[pivot]:
                            array[cur], array[border] = array[border], array[cur]
                            border += 1

                    array[border], array[pivot] = array[pivot], array[border]
                    return border

                while start <= end:
                    p = partition(start, end)
                    if p == k_largest:
                        break
                    elif p > k_largest:
                        end = p - 1
                    else:
                        start = p + 1

            def topKFrequent(self, nums: List[int], k: int) -> List[int]:
                if not nums or k <= 0:
                    return []

                counter = collections.Counter(nums)

                key_cnt_arr = []
                for key, cnt in counter.items():
                    key_cnt_arr.append(Element(key, cnt))

                self.quick_select(array=keykey_cnt_arr_cnt,
                                  start=0,
                                  end=len(key_cnt_arr)-1,
                                  k_largest=k-1)

                return [e.key for e in key_cnt_arr[:k]]
            ```
        * Approach4: bucket-sort, Time:O(n), Space:O(n)
          * Since we know the max freq would be len(nums), we can use bucket sort
          * Complexity:
            * Space:
              * Counter:O(n)
              * Bucket:O(n)
            * Time: O(n)
          * Python
            ```python
            def topKFrequent(self, nums: List[int], k: int) -> List[int]:

              if not nums or k <= 0:
                  return []

              """
              1. Create the character counter
              """
              counter = collections.Counter(nums)

              """
              1. Append the character to the bucket according to the cnt
              """
              freq_bucket = [None] * (len(nums) + 1)
              for key, cnt in counter.items():
                  # lazy creation
                  if not freq_bucket[cnt]:
                      freq_bucket[cnt] = []
                  freq_bucket[cnt].append(key)

              """
              1. Create the top k response
              """
              top_k = []
              for cnt in range(len(nums), 0, -1):
                  if not freq_bucket[cnt]:
                      continue

                  top_k.extend(freq_bucket[cnt])

                  if len(top_k) >= k:
                      break

              return top_k
            ```
    * 692: **Top K Frequent** Words (M)
      * Description:
        * Given a non-empty list of words, return the k most frequent elements.
        * Your answer should be sorted by frequency from highest to lowest. I**f two words have the same frequency, then the word with the lower alphabetical order comes first**.
      * Note:
        * notice the alphabetical order
      * Approach1: min heap, Time:O(nlogk), Space:O(n):
        * notice the pop order of min_heap
        * Complexity:
          * Space:
            * Counter: O(n)
            * Heap:O(k)
          * Time: O(n) + O(nlogk)
            * Create Counter: O(n)
            * Insert to Heap: O(nlogk)
            * Pop from heap: O(klogk)
        * Python:
          ```python
          import collections

          class Element(object):

              __slots__ = ['key', 'cnt']

              def __init__(self, key, cnt):
                  self.key = key
                  self.cnt = cnt

              def __lt__(self, other):
                  """
                  min heap
                  if the cnt is different, pop the smaller cnt first
                  else pop high alphabetical first
                  """
                  if self.cnt == other.cnt:
                      # pop high alphabetical first
                      return self.key > other.key
                  else:
                      return self.cnt < other.cnt

          class Solution:
              def topKFrequent(self, words: List[str], k: int) -> List[str]:
                  counter = collections.Counter(words)

                  min_heap = []
                  for key, cnt in counter.items():
                      e = Element(key, cnt)
                      if len(min_heap) < k:
                          heapq.heappush(min_heap, e)
                      elif e > min_heap[0]:
                          heapq.heapreplace(min_heap, e)

                  res = collections.deque()
                  while min_heap and len(res) < k:
                      e = heapq.heappop(min_heap)
                      res.appendleft(e.key)

                  return res
          ```
      * Approach2: bucket sort, Time:O(n), Space:O(n)
        * Each bucket stores a trie structure with different freq.
          * use trie to handle alphabetical order
        * Ref:
          * https://leetcode.com/problems/top-k-frequent-words/discuss/108399/Java-O(n)-solution-using-HashMap-BucketSort-and-Trie-22ms-Beat-81
      * Quick select is not suitable due to the output should be alphabetical order.
    * 451: Sort Characters By Frequency
      * This is similar to 347: **Top K Frequent** Elements
      * Complexity:
        * Time:
          * average cost of heapify if O(n)
      * Approach1: Sorting, Time:O(nlogn), Space:O(logn~n)
      * Approach2: min-heap, Time:O(n), Space:O(n)
        * Python
          ```python
          class Element(object):

              __slots__ = ['key', 'cnt']

              def __init__(self, key, cnt):
                  self.key = key
                  self.cnt = cnt

              def __lt__(self, other):
                  return self.cnt < other.cnt

          def frequencySort(self, s: str) -> str:
              """
              1. create freq cnt for each char
              """
              freq_cnt = collections.Counter(s)

              """
              1. heapify (linear time)
              """
              min_heap = [Element(key, cnt) for key, cnt in freq_cnt.items()]
              heapq.heapify(min_heap)

              """
              1. pop from min-heap (use deque to prevent reverse operation)
              """
              res = collections.deque()
              while min_heap:
                  e = heapq.heappop(min_heap)
                  res.appendleft(e.key * e.cnt)

              return ''.join(res)
          ```
      * Approach3: bucket-sort, Time:O(n), Space:O(n)
        * Since we know the max freq is len(s), bucket sort is suitable.
        * Python
          ```python
          class Element(object):
            __slots__ = ['key', 'cnt']
            def __init__(self, key, cnt):
                self.key = key
                self.cnt = cnt

            def __lt__(self, other):
                return self.cnt < other.cnt

          def frequencySort(self, s: str) -> str:
            """
            1: create the frequency map for each char
            """
            freq_cnt = collections.Counter(s)

            """
            1. The chars in the same bucket have the same frequency.
            """
            bucket = [None] * (len(s) + 1)
            for key, cnt in freq_cnt.items():
                if not bucket[cnt]:
                    bucket[cnt] = [key]
                else:
                    bucket[cnt].append(key)

            """
            1. pop char from high freq to low freq and format the string
            """
            res = []
            for cnt in range(len(s), 0, -1):
                if not bucket[cnt]:
                    continue
                for key in bucket[cnt]:
                    res.append(cnt * key)

            return ''.join(res)
          ```
      * quick-select can not apply this problem since the output order.
    * 973: K Closest Points to Origin
      * Description:
      * Approach1: Sorting, Time:O(nlogn), Space:O(logn~n)
      * Approach2: Max-heap, Time:O(nlogk), Space:O(k)
        * Python
          ```python
          class Element(object):

            __slots__ = 'point'

            @staticmethod
            def distance(x, y):
                """
                Euclidean distance to the origin (0,0)
                """
                return x**2 + y **2

            def __init__(self, point):
                # coordinate tuple (x, y)
                self.point = point

            def __eq__(self, other):
                return self.distance(*self.point) == self.distance(*other.point)

            def __lt__(self, other):
                """
                reverse the operator for min heap
                """
                return self.distance(*self.point) > self.distance(*other.point)

            def __gt__(self, other):
                """
                reverse the operator for min heap
                """
                return self.distance(*self.point) < self.distance(*other.point)

          def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
              max_heap = []
              for p in points:
                  e = Element(p)
                  #print(e, e.distance(*e.point))
                  if len(max_heap) < K:
                      heapq.heappush(max_heap, e)
                  elif e > max_heap[0]:
                      """
                      heapq is min heap, reverse the operator
                      """
                      heapq.heapreplace(max_heap, e)

              return [e.point for e in max_heap]
            ```
      * Approach3: Quick select, Time:O(n), Space:O(n)
        * Python
          ```python
          class Element(object):

            __slots__ = 'point'

            @staticmethod
            def distance(x, y):
                """
                Euclidean distance to the origin (0,0)
                """
                return x**2 + y **2

            def __init__(self, point):
                # coordinate tuple (x, y)
                self.point = point

            def __le__(self, other):
                return self.distance(*self.point) <= self.distance(*other.point)

          class Solution:
              def partition(self, array, start, end):
                  ran = random.randint(start, end)
                  pivot = end
                  array[pivot], array[ran] = array[ran], array[pivot]

                  border = start
                  for cur in range(start, end):
                      if array[cur] <= array[pivot]:
                          array[cur], array[border] = array[border], array[cur]
                          border += 1

                  array[border], array[pivot] = array[pivot], array[border]
                  return border

              def quick_select(self, array, start, end, k_smallest):
                  while start <= end:
                      p = self.partition(array, start, end)
                      if p == k_smallest:
                          break
                      elif p < k_smallest:
                          start = p + 1
                      else:
                          end = p - 1

              def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
                  array = [Element(p) for p in points]
                  self.quick_select(array=array,
                                    start=0,
                                    end=len(array)-1,
                                    k_smallest=K-1)

                  res = []
                  for e in array:
                      res.append(e.point)
                      if len(res) == K:
                          break

                  return res
            ```
  * Stream
    * 703: **Kth Largest** Element in a Stream
    * 295: Find **Median** from Data Stream (H)
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
                    pop = -heapq.heappushpop(self.lo, -num)
                    heapq.heappush(self.hi, pop)

            def findMedian(self) -> float:

                if len(self.lo) == len(self.hi):
                    return  (-self.lo[0] + self.hi[0]) / 2
                else:
                    return -self.lo[0]
        ```
  * 480: Sliding Window Median (H)
    * Approach1: Sorting, Time:O(nk)
      * Keep the sorted window with size k, for each iteration
        * Insert the new element (keep the window sorted)
          * O(k)
        * Remove the out of window element (keep the window sorted)
          * O(k)
        * Complexity would be O(nk)
    * Approach2: Two Heaps (Lazy Removal)
      * The idea is the same as Find **Median** from Data Stream.
      * The only additional requirement is removing the outgoing elements from the window.
        * At this point, an important thing to notice is the fact that if the two heaps are balanced, only the top of the heaps are actually needed to find the medians. This means that as long as we can somehow keep the heaps balanced, we could also keep some extraneous elements.
        * **Thus, we can use hash-tables to keep track of invalidated elements. Once they reach the heap tops, we remove them from the heaps. This is the lazy removal technique.**
      * Ref:
        * question, how to keep balanced ??
        * https://leetcode.com/problems/sliding-window-median/discuss/262689/Python-Small-and-Large-Heaps
  * 313: Super Ugly Numbe (M)
  * 218: The Skyline Problem (H)
    * Ref:
      * https://briangordon.github.io/2014/08/the-skyline-problem.html
### Cache
  * 146: LRU Cache (M)
    * Note:
      * Don't forget to update val of key in put operation.
    * Approach1: ordered dict
    * Approach2: dict + doubly linked list
      * Data Structures
        * **self.nodes**
          * Each key is mapping to the corresponding node, where we can retrieve the node in O(1) time.
        * **self.list**
          * DLinkedList keeps the nodes with LRU order (head node is the most recently used)
      * Python
        ```python
        class DLinkedNode(object):

          __slots__ = ['key', 'val', 'prev', 'next']

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

          def append_to_head(self, new):
              new.prev = self.head
              new.next = self.head.next

              self.head.next.prev = new
              self.head.next = new

          def pop_from_tail(self):
              if self.tail.prev is self.head:
                  return None

              pop = self.tail.prev
              self._remove_node(pop)
              return pop

          def _remove_node(self, node):
              node.prev.next = node.next
              node.next.prev = node.prev

          def move_to_head(self, node):
              self._remove_node(node)
              self.append_to_head(node)


      class LRUCache(object):

          def __init__(self, capacity):
              """
              :type capacity: int
              """
              self.nodes = dict()
              self.dlist = DLinkedList()
              self.cap = capacity
              self.len = 0

          def get(self, key):
              """
              :type key: int
              :rtype: int
              """
              if key not in self.nodes:
                  return -1

              node = self.nodes[key]
              self.dlist.move_to_head(node)
              return node.val

          def put(self, key, value):
              """
              :type key: int
              :type value: int
              :rtype: None
              """
              if self.cap < 1:
                  return

              if key in self.nodes:
                  node = self.nodes[key]
                  node.val = value
                  self.dlist.move_to_head(node)
              else:
                  if self.len == self.cap:
                      pop_node = self.dlist.pop_from_tail()
                      self.nodes.pop(pop_node.key)
                      self.len -= 1

                  new = DLinkedNode(key, value)
                  self.nodes[key] = new
                  self.dlist.append_to_head(new)
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
          * Each frequency is mapped to a DLinkedList.
          * All nodes in the DLinkedList have the same frequency, freq.
          * Each * DLinkedList keeps the nodes with LRU order (head node is the most recently used)
        * **self.min_freq**
          * A minimum frequency is maintained to keep track of the minimum frequency of across all nodes in this cache, such that **the DLinkedList with the min frequency can always be retrieved in O(1) time**.
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

            __slots__ = ['key', 'val', 'prev', 'next', 'freq']

            def __init__(self, key=None, val=None, freq=None):
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

            def __len__(self):
                    return self.len

            def append_to_head(self, new):
                new.prev = self.head
                new.next = self.head.next

                self.head.next.prev = new
                self.head.next = new
                self.len += 1

            def pop_from_tail(self):
                if self.tail.prev is self.head:
                    return None

                pop = self.tail.prev
                self.remove_node(pop)
                return pop

            def remove_node(self, node):
                node.prev.next = node.next
                node.next.prev = node.prev
                self.len -= 1


        class LFUCache(object):

            def __init__(self, capacity):
                self.nodes = dict()
                self.freq_lists = collections.defaultdict(DLinkedList)
                self.min_freq = 0
                self.cap = capacity
                self.len = 0

            def _update(self, node):
                # 1. remove node from old linked list
                prev_list = self.freq_lists[node.freq]
                prev_list.remove_node(node)

                # 2. update min_freq if lengh of prev_list is 0
                if self.min_freq == node.freq and len(prev_list) == 0:
                    self.min_freq += 1

                # 3. increase freq and append the node to the new list
                node.freq += 1
                self.freq_lists[node.freq].append_to_head(node)

            def get(self, key):
                if key not in self.nodes:
                    return -1

                node = self.nodes[key]
                self._update(node)
                return node.val

            def put(self, key, value):
                if self.cap < 1:
                    return

                if key in self.nodes:
                    node = self.nodes[key]
                    node.val = value
                    self._update(node)

                else:
                    # 1. pop node from linked list with min frequency if necessary
                    if self.len == self.cap:
                        pop = self.freq_lists[self.min_freq].pop_from_tail()
                        self.nodes.pop(pop.key)
                        self.len -= 1

                    # 2. add new node
                    freq = 1
                    new = DLinkedNode(key, value, freq)
                    self.nodes[key] = new
                    self.freq_lists[freq].append_to_head(new)

                    # 3. update freq and len
                    self.min_freq = freq
                    self.len += 1
        ```
### Tree
  * **Preorder**
    * Traversal
      * 144: Binary Tree Preorder **Traversal** (M)
        * Approach1: Recursive, Time:O(n), Space:O(n):
          * Python:
            ```python
            def preorderTraversal(self, root: TreeNode) -> List[int]:
              def _preorder(node):
                  if not node:
                      return

                  visited.append(node.val)
                  _preorder(node.left)
                  _preorder(node.right)

              visited = []
              _preorder(root)
              return visited
            ```
        * Approach2: Iterative1, Time:O(n), Space:O(n):
          * Python
            ```python
            def preorderTraversal(self, root: TreeNode) -> List[int]:
                if not root:
                    return []

                visited = list()
                stack = [root]
                while stack:
                    node = stack.pop()
                    visited.append(node.val)

                    if node.right:
                        stack.append(node.right)

                    if node.left:
                        stack.append(node.left)

                return visited
            ```
        * Approach3: Iterative2, Time:O(n), Space:O(n):
          * Python
            ```python
            def preorderTraversal(self, root: TreeNode) -> List[int]:
              if not root:
                  return []

              stack = []
              visited = []

              cur = root
              while cur or stack:
                  if not cur:
                      cur = stack.pop()

                  visited.append(cur.val)
                  if cur.right:
                      stack.append(cur.right)
                  cur = cur.left

              return visited
            ```
      * 589: N-ary Tree Preorder Traversal (E)
        * Approah1: Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def preorder(self, root: 'Node') -> List[int]:
              def _preorder(node):
                  visited.append(node.val)
                  for child in node.children:
                      _preorder(child)

              if not root:
                  return []

              visited = []
              _preorder(root)
              return visited
            ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def preorder(self, root: 'Node') -> List[int]:
              if not root:
                  return []

              visited = []
              stack = [root]

              while stack:
                  node = stack.pop()
                  visited.append(node.val)
                  for i in range(len(node.children)-1, -1, -1):
                      stack.append(node.children[i])

              return visited
            ```
      * 114: **Flatten** Binary Tree to Linked List	(M)
        * Preorder solution:
          * Each node's right child points to the next node of a pre-order traversal.
        * Postorder solution
          * Right->Left->Root
        * Approach1: PreOrder: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def flatten(self, root: TreeNode) -> None:
              if not root:
                  return

              s = [root]
              while s:
                  node = s.pop()

                  if node.right:
                      s.append(node.right)

                  if node.left:
                      s.append(node.left)

                  if s:
                      node.right = s[-1]

                  node.left = None
            ```
        * Approach2: PostOrder, Recursive, Time:O(n), Space:O(n)
          * https://leetcode.com/problems/flatten-binary-tree-to-linked-list/submissions/
          * Python
            ```python
            def flatten(self, root: TreeNode) -> None:
              def dfs(node):
                  if not node:
                      return

                  dfs(node.right)
                  dfs(node.left)

                  """
                  prev pointers to previous flattern list
                  """
                  nonlocal prev
                  node.right = prev
                  node.left = None
                  prev = node


              if not root:
                  return

              prev = None
              dfs(root)
            ```
        * Approach3: PostOrder, Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def flatten(self, root: TreeNode) -> None:
              if not root:
                  return

              cur = root
              s = []
              prev = None
              while cur or s:
                  if cur:
                      s.append(cur)
                      cur = cur.right
                  else:
                      top = s[-1]
                      if top.left and top.left is not prev:
                          cur = top.left
                      else:
                          node = s.pop()
                          node.right = prev
                          node.left = None
                          prev = node
            ```
    * Same tree
      * 100: **Same** Tree (E)
        * Description:
          * Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants.
        * Approach1: Recursive Time:O(n), Space:O(n)
          * Python
            ```python
            def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
              def dfs(n1, n2):
                  if not n1 and not n2:
                      return True

                  if not n1 or not n2 or n1.val != n2.val:
                      return False

                  return dfs(n1.left, n2.left) and dfs(n1.right, n2.right)

              return dfs(p, q)
            ```
        * Approach2: Iterative Time:O(n), Space:O(n)
            * Python
            ```python
            def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
              is_same = True
              stack = [(p, q)]
              while stack:
                  n1, n2 = stack.pop()

                  if not n1 and not n2:
                      continue

                  if not n1 or not n2 or n1.val != n2.val:
                      is_same = False
                      break

                  # n1.val == n2.val
                  stack.append((n1.left, n2.left))
                  stack.append((n1.right, n2.right))

              return is_same
            ```
      * 572: Subtree of Another Tree (E)
        * Approach1: Recursive: Time:O(mn), Space:O(m+n)
          * Python
            ```python
            def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
              def is_same_tree(n1, n2):
                  if not n1 and not n2:
                      return True

                  if not n1 or not n2 or n1.val != n2.val:
                      return False

                  return is_same_tree(n1.left, n2.left) and is_same_tree(n1.right, n2.right)

              def is_sub_tree(s, t):
                  if not s and not t:
                      return True

                  if not s or not t:
                      return False

                  if is_same_tree(s, t):
                      return True

                  if is_sub_tree(s.left, t):
                      return True

                  if is_sub_tree(s.right, t):
                      return True

                  return False

              return is_sub_tree(s, t)
            ```
        * Approach2: Iterative: Time:O(mn), Space:O(m+n)
          * Python
            ```python
            def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
              def is_same_tree(n1, n2):
                  is_same = True
                  stack = [(n1, n2)]
                  while stack:
                      n1, n2 = stack.pop()

                      if not n1 and not n2:
                          continue

                      if not n1 or not n2 or n1.val != n2.val:
                          is_same = False
                          break

                      stack.append((n1.left, n2.left))
                      stack.append((n1.right, n2.right))

                  return is_same

              if not s and not t:
                  return True

              if not s or not t:
                  return False

              is_sub_tree = False
              stack = [s]
              while stack:
                  node = stack.pop()
                  if is_same_tree(node, t):
                      is_sub_tree = True
                      break

                  if node.left:
                      stack.append(node.left)

                  if node.right:
                      stack.append(node.right)

              return is_sub_tree
            ```
      * 101: Symmetric Tree (E)
        * Approach1: Recursive:
          * Python
            ```python
            def isSymmetric(self, root: TreeNode) -> bool:
              def dfs(n1, n2):
                  if not n1 and not n2:
                      return True

                  if not n1 or not n2 or n1.val != n2.val:
                      return False

                  return dfs(n1.left, n2.right) and dfs(n1.right, n2.left)

              if not root:
                  return True

              return dfs(root.left, root.right)
            ```
        * Approach2: Iterative:
          * Python
            ```python
            def isSymmetric(self, root: TreeNode) -> bool:

              if not root:
                  return True

              is_symmetric = True

              stack = [(root.left, root.right)]
              while stack:
                  n1, n2 = stack.pop()


                  if not n1 and not n2:
                      continue

                  if not n1 or not n2 or n1.val != n2.val:
                      is_symmetric = False
                      break

                  stack.append((n1.left, n2.right))
                  stack.append((n1.right, n2.left))

              return is_symmetric
            ```
    * Path
      * 257: Binary Tree Paths (E)
        * Description:
          * Given a binary tree, return all **root-to-leaf** paths.
        * Approach1: Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def binaryTreePaths(self, root: TreeNode) -> List[str]:
              def dfs(node, cur):
                  cur.append(str(node.val))

                  if not node.left and not node.right:
                      paths.append('->'.join(cur))
                  else:
                      if node.left:
                          dfs(node.left ,cur)

                      if node.right:
                          dfs(node.right ,cur)

                  cur.pop()

              if not root:
                  return []

              paths = []
              cur = []
              dfs(root, cur)
              return paths
            ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def binaryTreePaths(self, root: TreeNode) -> List[str]:
              if not root:
                  return []

              paths = []
              cur = []
              stack = [(root, False)]
              while stack:
                  node, backtrack = stack.pop()
                  if backtrack:
                      cur.pop()
                      continue

                  cur.append(str(node.val))
                  stack.append((None, True))
                  if not node.left and not node.right:
                      paths.append('->'.join(cur))
                      continue

                  if node.right:
                      stack.append((node.right, False))
                  if node.left:
                      stack.append((node.left, False))

              return paths
            ```
      * 129: Sum Root to Leaf Numbers (M)
        * Approach1: Recursive:
          * Python
            ```python
            def sumNumbers(self, root: TreeNode) -> int:
              def dfs(node, cur_sum):

                  cur_sum = cur_sum * 10 + node.val

                  if not node.left and not node.right:
                      nonlocal sum_leaf
                      sum_leaf += cur_sum
                      return

                  if node.left:
                      dfs(node.left, cur_sum)

                  if node.right:
                      dfs(node.right, cur_sum)

              if not root:
                  return 0

              sum_leaf = 0
              dfs(root, 0)
              return sum_leaf
            ```
        * Approach2: Iterative:
          * Python
            ```python
            def sumNumbers(self, root: TreeNode) -> int:
              if not root:
                  return 0

              stack = [(root, 0)]
              sum_leaf = 0
              while stack:
                  node, cur_sum = stack.pop()
                  cur_sum = cur_sum * 10 + node.val

                  if not node.left and not node.right:
                      sum_leaf += cur_sum
                      continue

                  if node.left:
                      stack.append((node.left, cur_sum))

                  if node.right:
                      stack.append((node.right, cur_sum))

              return sum_leaf
            ```
      * 112: Path Sum (E)
        * Description:
          * determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
        * Approach1: Recursive:
          * Python
            ```python
            def hasPathSum(self, root: TreeNode, sum: int) -> bool:
              def dfs(node, target):
                  nonlocal find_path_sum

                  if find_path_sum:
                      return

                  target -= node.val

                  if not node.left and not node.right and target == 0:
                      find_path_sum = True
                      return

                  if node.left:
                      dfs(node.left, target)

                  if node.right:
                      dfs(node.right, target)

              if not root:
                  return False

              find_path_sum = False
              dfs(node=root, target=sum)
              return find_path_sum
            ```
        * Approach2: Iterative:
          * Python
            ```python
            def hasPathSum(self, root: TreeNode, sum: int) -> bool:
              if not root:
                  return False

              find_path_sum = False
              s = [(root, sum)]
              while s:
                  node, target = s.pop()
                  target -= node.val

                  if not node.left and not node.right and target == 0:
                      find_path_sum = True
                      break

                  if node.left:
                      s.append((node.left, target))

                  if node.right:
                      s.append((node.right, target))

              return find_path_sum

            ```
      * 113: Path Sum II (M)
        * Description:
          * Given a binary tree and a sum, find all **root-to-leaf** paths where each path's sum equals the given sum.
        * Approach1: Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
              def dfs(node, cur, target):
                  target -= node.val
                  cur.append(node.val)
                  if not node.left and not node.right and target == 0:
                      path_sum_comb.append(cur[:])
                  else:
                      if node.left:
                          dfs(node.left, cur, target)

                      if node.right:
                          dfs(node.right, cur, target)
                  cur.pop()

              if not root:
                  return []

              path_sum_comb = []
              dfs(root, [], sum)

              return path_sum_comb
            ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
              if not root:
                  return []

              path_sum_comb = []
              cur = []
              s = [(root, sum, False)]
              while s:
                  node, target, backtrack = s.pop()
                  if backtrack:
                      cur.pop()
                      continue

                  target -= node.val
                  cur.append(node.val)
                  s.append((None, None, True))

                  if not node.left and not node.right and target == 0:
                      path_sum_comb.append(cur[:])
                      continue

                  if node.left:
                      s.append((node.left, target, False))

                  if node.right:
                      s.append((node.right, target, False))

              return path_sum_comb
            ```
      * 437: Path Sum III (M)
        * Description:
          * The path does not need to start or end at the root or a leaf, **but it must go downwards** (traveling only from parent nodes to child nodes).
        * This is similar to 560: Subarray Sum Equals K
        * Approach1: Recursive:
          * Python
            ```python
            def pathSum(self, root: TreeNode, sum: int) -> int:
              def dfs(node, cur_sum, sum_d):
                  cur_sum += node.val
                  """
                  prev_sum + target = cur_sum
                  try to find cur_sum - target in the dictionary
                  """
                  nonlocal path_sum_cnt
                  path_sum_cnt += sum_d[cur_sum - target]
                  sum_d[cur_sum] += 1

                  if node.left:
                      dfs(node.left, cur_sum, sum_d)

                  if node.right:
                      dfs(node.right, cur_sum, sum_d)

                  """
                  backtracking
                  """
                  sum_d[cur_sum] -= 1

              if not root:
                  return 0

              target = sum
              path_sum_cnt = 0
              sum_d = collections.defaultdict(int)
              sum_d[0] = 1
              dfs(root, 0, sum_d)
              return path_sum_cnt
            ```
        * Approach2: Iterative:
          * Python
            ```python
            def pathSum(self, root: TreeNode, sum: int) -> int:
              if not root:
                  return 0

              target = sum
              path_sum_cnt = 0
              sum_d = collections.defaultdict(int)
              sum_d[0] = 1

              stack = [(root, 0, False)]
              while stack:
                  node, cur_sum, backtrack = stack.pop()
                  if backtrack:
                      sum_d[cur_sum] -= 1
                      continue

                  cur_sum += node.val
                  # prev_sum + target = cur_sum
                  path_sum_cnt += sum_d[cur_sum - target]

                  sum_d[cur_sum] += 1
                  stack.append((None, cur_sum, True))
                  if node.left:
                      stack.append((node.left, cur_sum, False))

                  if node.right:
                      stack.append((node.right, cur_sum, False))

              return path_sum_cnt
            ```
      * 988: Smallest String Starting From Leaf (M)
        * Description:
          * Find the lexicographically smallest string that **starts at a leaf of this tree and ends at the root**.
        * d is the avg depth of the tree, n is the number of the nodes
        * Approach1: Recursive, Time:O(dn), Space:O(n)
          * Python
            ```python
            def smallestFromLeaf(self, root: TreeNode) -> str:
              def dfs(node, cur):
                  if not node:
                      return

                  cur.appendleft(chr(node.val + ord_a))
                  if not node.left and not node.right:
                      nonlocal min_str
                      min_str = min(min_str, "".join(cur))

                  else:
                      dfs(node.left, cur)
                      dfs(node.right, cur)

                  cur.popleft()

              if not root:
                  return ""

              min_str = "~"
              ord_a = ord('a')
              cur = collections.deque()
              dfs(root, cur)
              return min_str
            ```
        * Approach2: Iterative, Time:O(dn), Space:O(n)
          * Python
            ```python
            def smallestFromLeaf(self, root: TreeNode) -> str:
                if not root:
                    return ""

                ord_a = ord('a')
                path = collections.deque()
                Event = collections.namedtuple('Event', ['node', 'backtrack'])
                stack = [Event(node=root, backtrack=False)]
                # ~ is greater than 26
                min_str = "~"
                while stack:

                    e = stack.pop()
                    node, backtrack = e.node, e.backtrack

                    if backtrack:
                        path.popleft()
                        continue

                    # number fo chr
                    path.appendleft(chr(node.val + ord_a))

                    # for backtrack
                    stack.append(Event(node=None, backtrack=True))

                    if not node.left and not node.right:
                        min_str = min(min_str, "".join(path))
                        continue

                    if node.left:
                        stack.append(Event(node=node.left, backtrack=False))

                    if node.right:
                        stack.append(Event(node=node.right, backtrack=False))

                return min_str
            ```
    * 226. **Invert** Binary Tree (E)
      * Approach1: Recursive, Recursive, Time:O(n), Space:O(n)
          ```python
          def invertTree(self, root: TreeNode) -> TreeNode:
            def dfs(node):
                if not node:
                    return
                node.left, node.right = node.right, node.left
                dfs(node.left)
                dfs(node.right)

            dfs(root)
            return root
          ```
      * Approach2: Iterative, Time:O(n), Space:O(n)
          ```python
          def invertTree(self, root: TreeNode) -> TreeNode:
            if not root:
                return root

            stack = [root]
            while stack:
                node = stack.pop()
                node.left, node.right = node.right, node.left
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            return root
          ```
    * 111: **Minimum Depth** of Binary Tree	(E)
      * Description:
        * The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
        * **A leaf is a node with no children.**
      * So root can be the leaf node if it does not have children.
      * Approach1: Recursive, Top Down, Time:O(n), Space:O(n)
        * Python
          ```python
          def minDepth(self, root: TreeNode) -> int:
            def dfs(node, depth):
                if not node.left and not node.right:
                    nonlocal min_depth
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
      * Approach2: Recursive, Bottom up, Time:O(n), Space:O(n)
        * Python
          ```python
          def minDepth(self, root: TreeNode) -> int:
            def dfs(node):
                if not node:
                    # This val should not be included
                    return float('inf')

                # leaf node
                if not node.left and not node.right:
                    return 1

                return 1 + min(dfs(node.left), dfs(node.right))

            if not root:
                return 0

            return dfs(root)
          ```
      * Approach3: Iterative, Top Down, Time:O(n), Space:O(n)
        * Python
          ```python
          def minDepth(self, root: TreeNode) -> int:
            if not root:
                return 0

            stack = [(root, 1)]
            min_depth = float('inf')

            while stack:
                node, depth = stack.pop()

                if not node.left and not node.right:
                    min_depth = min(min_depth, depth)
                    continue

                if node.left:
                    stack.append((node.left, depth+1))

                if node.right:
                    stack.append((node.right, depth+1))

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
    * 617: Merge Two Binary Trees (E)
      * Description:
        * You need to **merge them into a new binary tree**. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.
      * Approach1: Recursive, Time:O(n), Space:O(n)
        * Python
          ```python
          def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
            def merge_tree(cur, t1, t2):
                l_t1 = r_t1 = None
                l_t2 = r_t2 = None

                if t1:
                    cur.val += t1.val
                    l_t1, r_t1 = t1.left, t1.right

                if t2:
                    cur.val += t2.val
                    l_t2, r_t2 = t2.left, t2.right

                if l_t1 or l_t2:
                    cur.left = TreeNode(0)
                    merge_tree(cur.left, l_t1, l_t2)

                if r_t1 or r_t2:
                    cur.right = TreeNode(0)
                    merge_tree(cur.right, r_t1, r_t2)

            if not t1 and not t2:
                return None

            root = TreeNode(0)
            merge_tree(root, t1, t2)
            return root
          ```
      * Approach2: Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
            if not t1 and not t2:
                return None

            root = TreeNode(0)
            s = [(root, t1, t2)]

            while s:
                cur, t1, t2 = s.pop()

                l_t1 = r_t1 = None
                l_t2 = r_t2 = None

                if t1:
                    cur.val += t1.val
                    l_t1, r_t1 = t1.left, t1.right

                if t2:
                    cur.val += t2.val
                    l_t2, r_t2 = t2.left, t2.right

                if l_t1 or l_t2:
                    cur.left = TreeNode(0)
                    s.append((cur.left, l_t1, l_t2))

                if r_t1 or r_t2:
                    cur.right = TreeNode(0)
                    s.append((cur.right, r_t1, r_t2))

            return root
          ```
      * Approach3: Recursive-2, Time:O(n), Space:O(n)
        * Note:
          * In python, 'and' operation will not return a boolean value, it depends on the 'values'. Some values like '0', '', '{}', '[]' and so on, will be judged by false. And other values will be judged by true. The whole sentence will be judged like a boolean value. If it's true, it will return the last true value, remember is the value, not True. Otherwise, it will return the first false value.
        * Python
          ```python
          def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
            def dfs(t1, t2):
                if not t1 and not t2:
                    return None

                cur = TreeNode(0)
                for src in (t1, t2):
                    if src:
                        cur.val += src.val

                """
                t1 and t1.left will return first true value not true
                """
                cur.left = dfs(t1 and t1.left, t2 and t2.left)
                cur.right = dfs(t1 and t1.right, t2 and t2.right)
                return cur

            return dfs(t1, t2)
          ```
    * 872: Leaf-Similar Trees (E)
  * **Inorder**
    * Traversal
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

              visited = []
              cur = root
              stack = []
              while cur or stack:
                  if cur:
                      stack.append(cur)
                      cur = cur.left
                  else:
                      node = stack.pop()
                      visited.append(node.val)
                      if node.right:
                          cur = node.right

              return visited
            ```
      * 173: Binary Search Tree **Iterator** (M) *
        * Approach1: In-Order Iterative
          * Python
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
    * Construct
      * 105: Construct Binary Tree from **Preorder** and **Inorder** Traversal (M)
        * Preorder -> find the root node (from left)
        * Inorder -> find the left subtree and right subtree of each root
        * Approach1: DFS Recursive (optimization), Time:O(n), Space:O(n)
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
                    val = preorder[pre_idx]
                    node = TreeNode(val)
                    pre_idx += 1
                    partition = inorder_d[val]

                    node.left = build_tree(left, partition-1)
                    node.right = build_tree(partition+1, right)

                    return node

                if not preorder or not inorder or len(preorder) != len(inorder):
                    return None

                inorder_d = dict()
                for idx, val in enumerate(inorder):
                    inorder_d[val] = idx

                pre_idx = 0
                return build_tree(0, len(inorder)-1)
              ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
              if not preorder or not inorder:
                  return None

              inorder_d = dict()
              for idx, val in enumerate(inorder):
                  inorder_d[val] = idx

              pre_idx = 0
              root = TreeNode(0)
              stack = [(root, 0, len(inorder)-1)]

              while stack:
                  node, left, right = stack.pop()
                  node.val = preorder[pre_idx]
                  partition = inorder_d[node.val]

                  pre_idx += 1 # control the cur root idx of preorder

                  """
                  build left subtree first
                  """
                  if partition < right:
                      node.right = TreeNode(0)
                      stack.append((node.right, partition+1, right))

                  if l < partition:
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

                  """
                  build right subtree first
                  """
                  if left <= partition-1:
                      node.left = TreeNode(0)
                      stack.append((node.left, left, partition-1))

                  if partition+1 <= right:
                      node.right = TreeNode(0)
                      stack.append((node.right, partition+1, right))

              return root
            ```
      * 108: Convert **Sorted Array** to Binary Search Tree	binary search (E)
        * Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
        * A **height-balanced binary** tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
        * Approach1: Recursive: Time:O(n), Space:O(log(n))
          * Python
            ```python
            def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
              def build_tree(left, right):
                  mid = (left + right) // 2
                  node = TreeNode(nums[mid])

                  if left <= mid - 1:
                      node.left = build_tree(left, mid-1)

                  if mid + 1 <= right:
                      node.right = build_tree(mid+1, right)

                  return node

              if not nums:
                  return None

              return build_tree(0, len(nums)-1)
            ```
        * Approach2: Iterative: Time:O(n), Space:(log(n))
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
    * Successor
      * The successor of a node p is the node with the smallest key greater than p.val.
      * 285: **Inorder Successor** in BST (M)
        * successor:
          * if the node with the smallest key greater than p.val
          * Successor is the smallest node in the inorder traversal after the current one.
        * Approach1: Inorder Traversal: Time:O(h), Space:O(h)
          * Python
            ```python
            def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
              def get_inorder(node):
                  cur = node
                  stack = []
                  while cur or stack:
                      if cur:
                          stack.append(cur)
                          cur = cur.left
                      else:
                          node = stack.pop()
                          yield node
                          cur = node.right

              if not root or not p:
                  return None

              is_found = False
              successor = None
              for node in get_inorder(root):
                  if is_found:
                      successor = node
                      break
                  if node is p:
                      is_found = True

              return successor
            ```
      * 510: **Inorder Successor** in BST II (M)
        * Two cases:
          * Have right subtree:
            * return minimum value of right subtree
          * Do not have right subtree
            * find the successor from ancestors
        * Approach1: Inorder Traversal, Time:O(h), Space:O(n)
          * See 285
        * Approach2: Use parent pointer: Time:O(h), Space:O(1)
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
  * **Postorder**
    * Traversal
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
        * Approach2: Iterative1, from end to start, Time:O(n), Space:O(n)
          * traverse order: root -> right -> left (use deque)
          * Python
            ```python
            def postorderTraversal(self, root: TreeNode) -> List[int]:
              if not root:
                  return []

              visited = collections.deque()
              stack = [root]

              while stack:
                  node = stack.pop()
                  visited.appendleft(node.val)
                  if node.left:
                      stack.append(node.left)
                  if node.right:
                      stack.append(node.right)

              return visited
            ```
        * Approach3: Iterative2, from start to end, Time:O(n), Space:O(n)
          * traverse order: left -> right -> root
          * Python
            ```python
            def postorderTraversal(self, root: TreeNode) -> List[int]:
              if not root:
                  return

              stack = []
              visited = []
              prev = None
              cur = root

              while cur or stack:
                  if cur:
                      stack.append(cur)
                      cur = cur.left
                  else:
                      top = stack[-1]
                      """
                      top.right != prev_traverse means top.right has not traversed yet
                      check if right subtree has been traversed.
                      """
                      if top.right and top.right is not prev:
                          cur = top.right

                      else:
                          visited.append(top.val)
                          prev = stack.pop()

              return visited
            ```
        * Approach4: Morris Traversal, Time:O(n), Space:O(1)
      * 590: N-ary Tree Postorder Traversal (E)
        * Approach1: Recursive:
          * Python
            ```python
            def postorder(self, root: 'Node') -> List[int]:
              def dfs(node):
                  for child in node.children:
                      dfs(child)

                  visited.append(node.val)

              if not root:
                  return []

              visited = []
              dfs(root)
              return visited
            ```
        * Approach2: Iterative:
          * Python
            ```python
            def postorder(self, root: 'Node') -> List[int]:
              if not root:
                  return []

              visited = collections.deque()
              stack = [root]

              while stack:
                  node = stack.pop()
                  visited.appendleft(node.val)

                  for child in node.children:
                      stack.append(child)

              return visited
            ```
    * Path
      * Definition:
        * A path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
        * A "path" is like unicursal, **can't come back**
      * 124: Binary Tree **Maximum Path Sum** (H)
        * Description:
          * Given a non-empty binary tree, find the maximum path sum.
        * There are two cases for each node:
          * case1: do not connect with parent, the node can connect with two children.
          * case2: connect with parent, the node can connect with at most one child.
        * Approach1: Iterative
          * Python
            ```python
            def maxPathSum(self, root: TreeNode) -> int:
              def postorder(node):
                  prev = None
                  cur = node
                  s = []
                  while cur or s:
                      if cur:
                          s.append(cur)
                          cur = cur.left
                      else:
                          top = s[-1]
                          if top.right and top.right != prev:
                              cur = top.right
                          else:
                              prev = top
                              yield s.pop()

              if not root:
                  return 0

              max_path_sum = float('-inf')
              memo = {None: 0}
              for node in postorder(root):

                  # 0 means do not connet to the subtree
                  l_p_s, r_p_s = max(0, memo[node.left]), max(0, memo[node.right])

                  # case1: do not connect with parent, the node can connect with two children.
                  max_path_sum = max(max_path_sum, node.val + l_p_s + r_p_s)

                  # case2: connect with parent, the node can connect with at most one child.
                  memo[node] = node.val + max(l_p_s, r_p_s)

              return max_path_sum
            ```
        * Approach2: Recursive
          * Python
            ```python
            def maxPathSum(self, root: TreeNode) -> int:
              def dfs(node):
                  if not node:
                      return 0

                  l_p_s, r_p_s = max(0, dfs(node.left)), max(0, dfs(node.right))
                  nonlocal max_path_sum
                  max_path_sum = max(max_path_sum, node.val + l_p_s + r_p_s)

                  return node.val + max(l_p_s, r_p_s)

              if not root:
                  return 0

              max_path_sum = float('-inf')
              dfs(root)
              return max_path_sum
            ```
      * 687: Longest **Univalue Path** (E)
        * Approach1: Recursive:
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
        * Approach2: Iterative:
          * Python
            ```python
            def longestUnivaluePath(self, root: TreeNode) -> int:
              def postorder(node):
                  prev = None
                  cur = node
                  s = []
                  while cur or s:
                      if cur:
                          s.append(cur)
                          cur = cur.left
                      else:
                          top = s[-1]
                          if top.right and top.right != prev:
                              cur = top.right
                          else:
                              prev = top
                              yield s.pop()

              if not root:
                  return 0

              max_lup = 1
              memo = dict()

              for node in postorder(root):

                  left_lup = right_lup = 0

                  if node.left and node.val == node.left.val:
                      left_lup = memo[node.left]

                  if node.right and node.val == node.right.val:
                      right_lup = memo[node.right]


                  max_lup = max(max_lup, 1 + left_lup + right_lup)

                  memo[node] = 1 + max(left_lup, right_lup)

              return max_lup - 1
            ```
      * 298: Binary Tree **Longest Consecutive** Sequence (M)
        * Description:
          * Given a binary tree, find the length of the longest consecutive sequence path.
          * The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
          * The longest consecutive path need to be from parent to child
        * Approach1: DFS Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def longestConsecutive(self, root: TreeNode) -> int:
              def dfs(node, lcp):
                  nonlocal max_lcp
                  max_lcp = max(max_lcp, lcp)
                  for child in (node.left, node.right):
                      if not child:
                          continue

                      if (node.val + 1) == child.val:
                          dfs(child, lcp+1)
                      else:
                          dfs(child, 1)

              if not root:
                  return 0

              max_lcp = 1
              dfs(root, 1)
              return max_lcp
            ```
        * Approach2-1: BFS Iterative, determine child's depth in the parent Time:O(n), Space:O(n)
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
        * Approach2-2: BFS Iterative, pass parent val, Time:O(n), Space:O(n)
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
      * 549: Binary Tree **Longest Consecutive** Sequence II (M)
        * Description:
          * Given a binary tree, you need to find the length of Longest Consecutive Path in Binary Tree.
          * On the other hand, **the path can be in the child-Parent-child order, where not necessarily be parent-child order**.
          * Especially, this path can be either **increasing or decreasing**.
        * 3 cases for each node:
          * 2 increasing subtree.
          * 2 decreasing subtree.
          * 1 increasing and 1 decreasing subtrees.
        * Approach1: Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def longestConsecutive(self, root: TreeNode) -> int:
              def dfs(node):
                  inc = dec = 1
                  for child in (node.left, node.right):
                      if not child:
                          continue
                      c_inc, c_dec = dfs(child)
                      if node.val + 1 == child.val:
                          inc = max(inc, 1 + c_inc)
                      elif node.val - 1 == child.val:
                          dec = max(dec, 1 + c_dec)

                  nonlocal max_lcp
                  """
                  For the case that do not connect to the parent
                  """
                  max_lcp = max(max_lcp, inc + dec - 1)

                  """
                  For the case that connects to the parent
                  return max inc and dec path for parent
                  """
                  return inc, dec

              if not root:
                  return 0

              max_lcp = 1
              dfs(root)
              return max_lcp
            ```
        * Approach2: Iterative: Time:O(n), Space:O(n)
          * Python
            ```python
            def longestConsecutive(self, root: TreeNode) -> int:
              def get_postorder(node):
                  postorder = collections.deque()
                  stack = [root]
                  while stack:
                      node = stack.pop()
                      postorder.appendleft(node)
                      if node.left:
                          stack.append(node.left)
                      if node.right:
                          stack.append(node.right)

                  return postorder

              if not root:
                  return 0

              postorder = get_postorder(root)
              max_lcp = 1
              memo = dict()
              for node in postorder:
                  inc = dec = 1
                  for child in (node.left, node.right):
                      if not child:
                          continue
                      c_inc, c_dec = memo[child]
                      # increasing path
                      if node.val + 1 == child.val:
                          inc = max(inc, 1 + c_inc)
                      # decreasing path
                      elif node.val - 1 == child.val:
                          dec = max(dec, 1 + c_dec)

                  """
                  For the case do not connect to the parent
                  """
                  max_lcp = max(max_lcp, inc + dec - 1)
                  """
                  For the case that connects to the parent/
                  """
                  memo[node] = (inc, dec)

              return max_lcp
            ```
    * 236: **Lowest Common Ancestor** of a **Binary Tree** (M)
      * Description:
        * The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants.
      * Algo:
        * https://leetcode.com/articles/lowest-common-ancestor-of-a-binary-tree/
      * Three cases need to consider:
        * mid + left
        * mid + right
        * left + right
      * Approach1: Recursive, Time:O(n), Space:O(n)
        * algo:
          * Start traversing the tree from the root node.
          * If either of the left or the right branch returns True, this means one of the two nodes was found below.
          * If the current node itself is one of p or q, we would mark a variable mid as True and continue the search for the other node in the left and right branches.
          * If at any point in the traversal, any two of the three flags left, right or mid become True, this means we have found the lowest common ancestor for the nodes p and q.
        * Python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            def dfs(node):
                nonlocal lca

                if not node or lca:
                    return False

                mid = False
                if node is p or node is q:
                    mid = True

                left = dfs(node.left)
                right = dfs(node.right)

                """
                case1: left + right
                case2: mid + left
                case3: mid + right
                """
                if mid + left + right == 2:
                    nonlocal lca
                    lca = node

                return mid or left or right

            lca = None
            dfs(root)
            return lca
          ```
      * Approach2: Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            def postorder(node):
                prev = None
                cur = node
                s = []
                while cur or s:
                    if cur:
                        s.append(cur)
                        cur = cur.left
                    else:
                        top = s[-1]
                        if top.right and top.right != prev:
                            cur = top.right
                        else:
                            prev = top
                            yield s.pop()

            lca = None
            memo = {None: False}
            for node in postorder(root):
                mid = False
                if node is p or node is q:
                    mid = True

                left, right = memo[node.left], memo[node.right]

                if mid + left + right == 2:
                    lca = node
                    break

                memo[node] = mid or left or right

            return lca
          ```
    * 110: **Balanced** Binary Tree (E)
      * Description
        * A binary tree in which the depth of the two subtrees of **every node** never differ by more than 1.
        * Even left subtree and right subtree are balanced trees, the tree may not be balanced tree.
      * Approach1: Recursive, Time:O(n), Space:O(n)
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

                  is_balanced = abs(left_d - right_d) <= 1
                  depth =  max(left_d, right_d) + 1
                  return is_balanced, depth

              if not root:
                  return True

              return dfs(root)[0]
            ```
      * Approach2: Iterative, Time:O(n), Space:O(n)
        * Python
          ```python
          def isBalanced(self, root: TreeNode) -> bool:
            def get_postorder(node):
                postorder = collections.deque()
                s = [root]
                while s:
                    node = s.pop()
                    postorder.appendleft(node)

                    if node.left:
                        s.append(node.left)
                    if node.right:
                        s.append(node.right)

                return postorder

            if not root:
                return True

            is_balanced = True
            postorder = get_postorder(root)
            memo = collections.defaultdict(int)
            for node in postorder:
                left_d, right_d = memo[node.left], memo[node.right]

                if abs(left_d - right_d) > 1:
                    is_balanced = False
                    break

                memo[node] = max(left_d, right_d) + 1

            return is_balanced
          ```
    * 366: **Find Leaves** of Binary Tree (M)
      * Approach1: recursive:
        * Python
          ```python
          def findLeaves(self, root: TreeNode) -> List[List[int]]:

            def dfs(node):
                if not node:
                    return 0

                depth = 1 + max(dfs(node.left), dfs(node.right))

                if len(leaves) < depth:
                    leaves.append([])

                leaves[depth-1].append(node.val)

                return depth


            if not root:
                return []

            leaves = []
            dfs(root)
            return leaves
          ```
      * Approach2: Iterative:
        * Python
          ```python
          def findLeaves(self, root: TreeNode) -> List[List[int]]:
            def get_postorder(node):
                postorder = collections.deque()
                s = [root]
                while s:
                    node = s.pop()
                    postorder.appendleft(node)
                    if node.left:
                        s.append(node.left)
                    if node.right:
                        s.append(node.right)

                return postorder

            if not root:
                return []

            leaves = []
            postorder = get_postorder(root)
            memo = {None: 0}
            for node in postorder:

                depth = 1 + max(memo[node.left], memo[node.right])

                if len(leaves) < depth:
                    leaves.append([])

                leaves[depth-1].append(node.val)

                memo[node] = depth

            return leaves
          ```
    * 250: Count **Univalue Subtrees** (M)
      * Description:
        * A Uni-value subtree means all nodes of the subtree have the same value.
        * subtree not path
      * Approach1: Recursive:
        * Python
          ```python
          def countUnivalSubtrees(self, root: TreeNode) -> int:
            def dfs(node):
                nonlocal uni_subtree
                # leaf node
                if not node.left and not node.right:
                    uni_subtree += 1
                    return True

                is_univalue = True
                for child in (node.left, node.right):
                    if not child:
                        continue
                    if not dfs(child) or node.val != child.val:
                        is_univalue = False
                        # can not break here, since subtree may be uni-value

                if is_univalue:
                    uni_subtree += 1
                return is_univalue

            if not root:
                return 0
            uni_subtree = 0
            dfs(root)
            return uni_subtree
          ```
      * Aooroch2: Iterative:
        * Python
          ```python
          def countUnivalSubtrees(self, root: TreeNode) -> int:
            def get_postorder(node):
                postorder = collections.deque()
                s = [root]
                while s:
                    node = s.pop()
                    postorder.appendleft(node)

                    if node.left:
                        s.append(node.left)
                    if node.right:
                        s.append(node.right)

                return postorder

            if not root:
                return 0

            uni_subtree = 0
            memo = dict()
            postorder = get_postorder(root)

            for node in postorder:
                if not node.left and not node.right:
                    uni_subtree += 1
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
                    uni_subtree += 1

                memo[node] = is_univalue

            return uni_subtree
          ```
    * 337: House Robber III (M)
      * See DP
  * **Level Order and BFS**
    * Traversal:
      * 102: Binary Tree Level Order Traversal (E)
        * Approach1: Recursive, Time:O(n), Space:O(n)
          * Python Solution
            ```python
            def levelOrder(self, root: TreeNode) -> List[List[int]]:
              def dfs(node, level):
                  if not node:
                      return

                  if len(traversal) < level + 1:
                      traversal.append([])

                  traversal[level].append(node.val)

                  dfs(node.left, level+1)
                  dfs(node.right, level+1)

              traversal = []
              dfs(root, 0)
              return traversal
            ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def levelOrder(self, root: TreeNode) -> List[List[int]]:
              if not root:
                  return []

              q = collections.deque([(root)])
              traversal = []

              while q:
                  cur_traversal = []
                  traversal.append(cur_traversal)

                  for _ in range(len(q)):
                      node = q.popleft()
                      cur_traversal.append(node.val)
                      if node.left:
                          q.append(node.left)
                      if node.right:
                          q.append(node.right)

              return traversal
            ```
      * 107: Binary Tree Level Order Traversal II (E)
        * Description:
          * eturn the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root.
        * Approach1: Recursive, Time:O(n), Space:O(n)
          * Same idea like iterative approach, use appendleft when createing new list
          ```python
            def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
              def dfs(node, level):
                  if not node:
                      return

                  if len(traversal) < level + 1:
                      traversal.appendleft([])

                  # len(traversal)-1-level also works
                  traversal[-(level+1)].append(node.val)

                  dfs(node.left, level+1)
                  dfs(node.right, level+1)

              traversal = collections.deque()
              dfs(root, 0)

              return traversal
          ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python Solution
            ```python
            def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
              if not root:
                  return []

              visits = collections.deque()
              q = collections.deque([root])

              while q:
                  cur_visits = []
                  visits.appendleft(cur_visits)
                  for _ in range(len(q)):
                      node = q.popleft()
                      cur_visits.append(node.val)

                      if node.left:
                          q.append(node.left)

                      if node.right:
                          q.append(node.right)

              return visits
            ```
      * 429: N-ary Tree Level Order Traversal (M)
        * Approach1: Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def levelOrder(self, root: 'Node') -> List[List[int]]:
              def dfs(node, level):
                  if len(visited) < level + 1:
                      visited.append([])

                  visited[level].append(node.val)
                  for child in node.children:
                      dfs(child, level+1)

              if not root:
                  return []

              visited = []
              dfs(root, level=0)
              return visited
            ```
        * Approach2: Iterative, Time:O(n), Space:O(n)
          * Python
            ```python
            def levelOrder(self, root: 'Node') -> List[List[int]]:
              if not root:
                  return []

              visited = []
              q = collections.deque([root])
              while q:
                  level_visited = []
                  visited.append(level_visited)
                  for _ in range(len(q)):
                      node = q.popleft()
                      level_visited.append(node.val)

                      for child in node.children:
                          q.append(child)

              return visited
            ```
      * 103: Binary Tree **Zigzag** Level Order Traversal (M)
        * Approach1: Iterative, Time:O(n), Space:O(n)
          * Python Solution:
            ```python
            def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
              if not root:
                  return []

              traversal = []
              q = collections.deque([root])
              append_left = False

              while q:
                  cur_traversal = collections.deque()
                  traversal.append(cur_traversal)
                  for _ in range(len(q)):
                      node = q.popleft()
                      if append_left:
                          cur_traversal.appendleft(node.val)
                      else:
                          cur_traversal.append(node.val)

                      if node.left:
                          q.append(node.left)
                      if node.right:
                          q.append(node.right)

                  append_left = not append_left

              return traversal
            ```
        * Approach2: Recursive, Time:O(n), Space:O(n)
          * Python Solution
            ```python
            def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
              def dfs(node, level):
                  if not node:
                      return

                  if len(traversal) < level + 1:
                      traversal.append(collections.deque())

                  if level % 2:
                      traversal[level].appendleft(node.val)
                  else:
                      traversal[level].append(node.val)

                  dfs(node.left, level+1)
                  dfs(node.right, level+1)

              traversal = []
              dfs(root, 0)
              return traversal
            ```
    * 637: Average of Levels in Binary Tree (E)
      * Description:
        * Given a non-empty binary tree, return the average value of the nodes on each level in the form of an array.
      * Approach1 Iterative: Time:O(n), Space:O(n)
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
      * Approach2 Recursive: Time:O(n), Space:O(n)
        * Python
          ```python
          def averageOfLevels(self, root: TreeNode) -> List[float]:
            def dfs(node, level):
                if not node:
                    return

                if len(avg_vals) < level + 1:
                    avg_vals.append([0, 0])

                avg_vals[level][SUM] += node.val
                avg_vals[level][CNT] += 1

                dfs(node.left, level+1)
                dfs(node.right, level+1)


            if not root:
                return []

            SUM, CNT = 0, 1
            avg_vals = []
            dfs(root, 0)

            return [float(e[0]/e[1]) for e in avg_vals]
          ```
    * 199: Binary Tree **Right Side** View (M)
      * Approach1 Iterative:
        * Python
          ```python
          def rightSideView(self, root: TreeNode) -> List[int]:
            if not root:
                return []

            right_side_view = []
            q = collections.deque([root])
            while q:
                node = None
                for _ in range(len(q)):
                    node = q.popleft()
                    if node.left:
                        q.append(node.left)

                    if node.right:
                        q.append(node.right)

                right_side_view.append(node.val)

            return right_side_view
          ```
      * Approach2 Recursive:
        * Python
          ```python
          def rightSideView(self, root: TreeNode) -> List[int]:
            def dfs(node, level):
                if not node:
                    return

                if len(right_side_view) < level + 1:
                    right_side_view.append(node.val)

                dfs(node.right, level+1)
                dfs(node.left, level+1)

            right_side_view = []
            dfs(root, 0)
            return right_side_view
          ```
    * 314: Binary Tree **Vertical Order** Traversal	(M)
      * Description:
        * Node in the same vertical order should be in the same group (from top to down)
        * List group from left to right
      * Approach1: Iterative, Time:O(nlogn), Space:O(n)
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
      * Approach2: Iterative, Time:O(n), Space:O(n)
        * Use min_idx and max_idx to prevent sorting
        * Python
          ```python
          def verticalOrder(self, root: TreeNode) -> List[List[int]]:
            if not root:
                return []

            d = collections.defaultdict(list)
            q = collections.deque([(root, 0)])
            min_idx = max_idx = 0
            while q:
                node, idx = q.popleft()
                d[idx].append(node.val)
                if node.left:
                    min_idx = min(min_idx, idx-1)
                    q.append((node.left, idx-1))

                if node.right:
                    max_idx = max(max_idx, idx+1)
                    q.append((node.right, idx+1))

            vertical_order = []
            for idx in range(min_idx, max_idx+1):
                vertical_order.append(d[idx])

            return vertical_order
          ```
    * 116 & 117: **Populating Next Right Pointers** in Each Node (M)
      * Description:
        * Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
      * A perfect binary tree
        * **where all leaves are on the same level, and every parent has two children.**
      * The key of space O(1) solution is use next pointer for level traversal.
      * Approach1: Use queue, Time:O(n), Space:O(n)
        * Python
          ```python
          def connect(self, root: 'Node') -> 'Node':
            if not root:
                return root

            q = collections.deque([root])
            dummy = Node(0)
            while q:
              prev = dummy
              for _ in range(len(q)):
                  cur = q.popleft()
                  if cur.left:
                      q.append(cur.left)
                  if cur.right:
                      q.append(cur.right)

                  prev.next = cur
                  prev = cur

            return root
          ```
      * Approach2: Use next pointer for level traversal, perfect binary tree, Time:O(n), Space:O(1)
        * Python
          ```python
          def connect(self, root: 'Node') -> 'Node':
            if not root:
                return None

            head = root

            while head and head.left:

                """
                level order traversal
                this solution is for perfect binary tree only
                """
                cur = head
                while cur:
                    cur.left.next = cur.right
                    cur.right.next = cur.next.left if cur.next else None
                    cur = cur.next

                head = head.left

            return root
          ```
      * Approach3: Use dummy and next pointer for level traversal, general solution, Time:O(n), Space:O(1)
        * Python
          ```python
          def connect(self, root: 'Node') -> 'Node':
            if not root:
              return None

            head = root
            dummy = Node(0)

            while head:
                """
                level order traversal
                """
                cur = head
                prev = dummy
                while cur:
                    if cur.left:
                        prev.next = cur.left
                        prev = prev.next

                    if cur.right:
                        prev.next = cur.right
                        prev = prev.next

                    cur = cur.next

                """
                dummy.next ponters to the head of next level
                """
                head = dummy.next
                dummy.next = None

            return root
          ```
  * Tree Slots:
    * 331: Verify Preorder Serialization of a Binary Tree	(M)
      * Approach1: Slots, Time:O(n), Space:O(n)
        * Python
          ```python
          def isValidSerialization(self, preorder: str) -> bool:
            if not preorder:
                return False

            # init for root
            is_valid = True
            slots = 1
            for node in preorder.split(','):

                if slots <= 0:
                    is_valid = False
                    break

                slots -= 1
                if node != '#':
                    slots += 2

            return is_valid and slots == 0
          ```
    * 297: Serialize and Deserialize Binary Tree (H)
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
                    return pickle.dumps([])

                bfs = []
                q = collections.deque([root])

                while q:
                    node = q.popleft()
                    if not node:
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
                bfs = pickle.loads(data)
                if not bfs:
                    return None

                root = TreeNode(bfs[0])
                i = 1
                q = collections.deque([root])
                while q:
                    node = q.popleft()
                    """
                    Each node has at least two slots
                    """
                    if bfs[i] != None:
                        node.left = TreeNode(bfs[i])
                        q.append(node.left)
                    i += 1

                    if bfs[i] != None:
                        node.right = TreeNode(bfs[i])
                        q.append(node.right)
                    i += 1

                return root
           ```
    * 428: Serialize and Deserialize N-ary Tree (H)
      * Approach1: Iterative
        * Python
          ```python
          class Codec:

            def serialize(self, root):
                if not root:
                    return pickle.dumps([])

                bfs = []
                bfs.append(root.val)
                q = collections.deque([root])
                """
                [1, 3, 3, 2, 4, 2, 5, 6, 0, 0, 0, 0]
                """
                while q:
                    node = q.popleft()
                    bfs.append(len(node.children))
                    for child in node.children:
                        bfs.append(child.val)
                        q.append(child)
                return pickle.dumps(bfs)

            def deserialize(self, data):
                bfs = pickle.loads(data)

                if not bfs:
                    return None

                root = Node(bfs[0], [])
                q = collections.deque([root])
                i = 1

                while q:
                    node = q.popleft()
                    child_cnt = bfs[i]
                    i += 1
                    for _ in range(child_cnt):
                        child = Node(bfs[i], [])
                        i += 1
                        node.children.append(child)
                        q.append(child)

                return root
          ```
  * **Binary Search Tree** (BST)
    * Definition of BST:
      * The **left subtree** of a node contains only nodes with keys **less than** the node's key.
      * The **right subtree** of a node contains only nodes with keys **greater than** the node's key.
      * Both the left and right subtrees must also be binary search trees.
    * Basic Operations
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
              def dfs(node, lo, hi):
                  nonlocal is_valid
                  if not lo < node.val < hi:
                      is_valid = False
                      return

                  if node.left:
                      dfs(node.left, lo, node.val)

                  if node.right:
                      dfs(node.right, node.val, hi)

              if not root:
                  return True

              is_valid = True
              dfs(root, float('-inf'), float('inf'))
              return is_valid
            ```
        * Approach3: Preorder Iterative, Time:O(h), Space:O(h)
          * Python
            ```python
            def isValidBST(self, root: TreeNode) -> bool:
              if not root:
                  return True

              is_valid = True
              stack = [(root, float('-inf'), float('+inf'))]

              while stack:
                  node, lo, hi = stack.pop()

                  if not lo < node.val < hi:
                      is_valid = False
                      break

                  if node.left:
                      stack.append((node.left, lo, node.val))

                  if node.right:
                      stack.append((node.right, node.val, hi))

              return is_valid
            ```
        * Approach4: InOrder Traversal, Time:O(h), Space:O(h)
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
      * 700: **Search** in a Binary Search Tree (E)
        * Approach1: Iterative, Time:O(n), Space:O(1)
          * Python
            ```python
            def searchBST(self, root: TreeNode, val: int) -> TreeNode:
              if not root:
                  return None

              target = None
              cur = root
              while cur:
                  if val == cur.val:
                      target = cur
                      break
                  elif val < cur.val:
                      cur = cur.left
                  else:
                      cur = cur.right

              return target
            ```
        * Approach2: Recursive, Time:O(n), Space:O(n)
          * Python
            ```python
            def searchBST(self, root: TreeNode, val: int) -> TreeNode:
              def dfs(node):
                  if not node:
                      return None

                  if val == node.val:
                      return node
                  elif val < node.val:
                      return dfs(node.left)
                  else:
                      return dfs(node.right)

              return dfs(root)
            ```
      * 701: **Insert** into a BST (M)
        * Recursive backtrack: Time:O(h), Space:O(h)
          * Python
            ```python
            def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
              def dfs(node):
                  if not node:
                      return TreeNode(val)

                  if val < node.val:
                      node.left = dfs(node.left)
                  else:
                      node.right = dfs(node.right)

                  return node

              if not root:
                  return TreeNode(val)

              dfs(root)
              return root
            ```
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
              while cur:
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
        * 2 Cases for parent of the target node:
          * No parent : the target node is root -> return new root
          * Having parent: the target node is not root -> return old root
        * 3 Cases for subtrees of the target node:
          * No subtrees: It is a leaf node.
          * It has only one subtree.
          * It has two subtrees.
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

              def get_right_min(node):
                  cur = node.right
                  while cur and cur.left:
                      cur = cur.left
                  return cur

              def delete_node(node):
                  # leaf node
                  if not node.left and not node.right:
                      return None

                  # no left subtree
                  if not node.left:
                      return node.right

                  # no right subtree
                  if not node.right:
                      return node.left

                  right_min = get_right_min(node)
                  right_min.left = node.left

                  # node.right replaces the node
                  return node.right

              def find_target_and_parent(node):
                  # find the target
                  parent = None
                  cur = node
                  while cur:
                      if key < cur.val:
                          parent = cur
                          cur = cur.left

                      elif key > cur.val:
                          parent = cur
                          cur = cur.right
                      else:
                          break

                  return parent, cur

                if not root:
                    return None

                parent, target = find_target_and_parent(root)

                # can not find the target node
                if not target:
                    return root

                # target node is the root (no parent)
                if not parent:
                    return delete_node(target)

                # target node has parent
                if key < parent.val:
                    parent.left = delete_node(target)
                else:
                    parent.right = delete_node(target)

                return root
            ```
    * Inorder (sorting)
      * 783: Minimum Distance Between BST Nodes
        * Approach1: Inorder Traversal, Time:O(n), Space:O(n)
          * Python
            ```python
            def minDiffInBST(self, root: TreeNode) -> int:
              def get_inorder(node):
                  cur = node
                  stack = []
                  while cur or stack:
                      if cur:
                          stack.append(cur)
                          cur = cur.left
                      else:
                          node = stack.pop()
                          yield node
                          cur = node.right

              if not root:
                  return 0

              min_diff = float('inf')
              prev = None
              for cur in get_inorder(root):
                  if prev:
                      min_diff = min(min_diff, cur.val - prev.val)
                  prev = cur

              return min_diff
            ```
      * 230: **Kth Smallest** Element in a BST (M)
        * Approach1: Inorder Traversal, Time:O(h+k), Space:O(h)
          * Python
            ```python
            def kthSmallest(self, root: TreeNode, k: int) -> int:
              def get_inorder(node):
                  cur = node
                  stack = []
                  while cur or stack:
                      if cur:
                          stack.append(cur)
                          cur = cur.left
                      else:
                          node = stack.pop()
                          yield node
                          cur = node.right

              cnt = 0
              target = None
              for node in get_inorder(root):
                  cnt += 1
                  if cnt == k:
                      target = node.val
                      break

              return target
            ```
      * Follow up:
        * What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kth Smallest routine?
        * Doubly Linked List + Dict + Tree ??
    * 235: **Lowest Common Ancestor** of a Binary Search Tree (E)
      * Description:
        * The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself)
        * All of the nodes' values will be unique.
        * p and q are different and both values will exist in the BST.
      * FAQ
        * How to error handling if p and q are not in the tree ?? Use 236 (postorder)
      * 3 cases:
        * lca in left subtree
        * lca in rght subtree
        * cur node is the lca (p in left, q in right or vice versa)
      * Preorder Recursive, Time:O(n), Space:O(n)
        * python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            def dfs(node):
                if not node:
                    return None

                val = node.val
                if p_val < val and q_val < val:
                    return dfs(node.left)
                elif p_val > val and q_val > val:
                    return dfs(node.right)
                else:
                    return node

            p_val = p.val
            q_val = q.val

            return dfs(root)
          ```
      * Preorder Iterative, Time:O(n), Space:O(1)
        * Python
          ```python
          def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            if not root or not p or not q:
                return None

            lca = None
            p_val, q_val = p.val, q.val
            cur = root
            while cur:
                val = cur.val
                if p_val < val and q_val < val:
                    cur = cur.left
                elif p_val > val and q_val > val:
                    cur = cur.right
                else:
                    lca = cur
                    break

            return lca
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
    * 449: Serialize and Deserialize BST
    * 333: Largest BST Subtree (M)
    * 530: Minimum Absolute Difference in BST
    * 938: Range Sum of BST (E)
    * 729: My Calendar I
    * 099: **Recover** Binary Search Tree (H)
  * TODO:
    * 156: Binary Tree Upside Down (M)
    * 652: Find Duplicate Subtrees
### Trie (Prefix Tree)
  * 208: Implement Trie (M)
    * Approach1: Iterative, Inert:O(k), Search:O(k)
      * Python:
        ```python
        class TrieNode(object):
          def __init__(self):
              self.children = collections.defaultdict(TrieNode)
              self.end_of_word = False

        class Trie:
            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.root = TrieNode()

            def insert(self, word: str) -> None:
                """
                Inserts a word into the trie.
                """
                if not word:
                    return

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
      * Python
        ```python
        class TrieNode(object):
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.end_of_word = False

        class Trie:

            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.root = TrieNode()

            def insert(self, word: str) -> None:
                """
                Inserts a word into the trie.
                """
                def insert_with_idx(cur, idx):
                    if idx == w_len:
                        cur.end_of_word = True
                        return

                    insert_with_idx(cur.children[word[idx]], idx+1)

                if not word:
                    return

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
                    if idx == p_len:
                        return True

                    c = prefix[idx]
                    if c not in cur.children:
                        return False

                    return search_with_idx(cur.children[c], idx+1)

                p_len = len(prefix)
                return search_with_idx(self.root, 0)
        ```
  * 211: Add and Search Word - Data structure design (M)
    * search word (**support wildcard**)
    * Approach1: Iterative:
      * Python
        ```python
        class TrieNode(object):
          def __init__(self):
              self.children = collections.defaultdict(Node)
              self.end_of_word = False

        class WordDictionary:
            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.root = TrieNode()

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
                        if not cur.end_of_word:
                          continue

                        found = True
                        break

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
      * Python
        ```python
        class TrieNode(object):
          def __init__(self):
              self.children = collections.defaultdict(Node)
              self.end_of_word = False

        class WordDictionary:
            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.root = TrieNode()

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
                        res = False
                        if c in cur.children:
                            res = search_with_idx(cur.children[c], idx+1)
                        return res

                w_len = len(word)
                return search_with_idx(self.root, 0)
        ```
  * 212: Word Search II (H)
  * 642: Design Search Autocomplete System (H)
  * 745: Prefix and Suffix Search (H)
### BFS & DFS
  * Note:
    * utlities:
  * 547: Friend Circles (M)
    * see union find
  * **Islands**
    * 200: Number of Islands (M)
      * Description:
        * Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
      * Approach1: BFS, Time: O(mn), Space: O(mn)
        * Set visits to True before append to the queue to **reduce unnecessary iterations.**
        * Python
          ```python
          DIRECTIONS = ((1, 0), (0, -1), (-1, 0), (0, 1))
          WATER = '0'
          LAND = '1'

          def numIslands(self, grid: List[List[str]]) -> int:
              def neighbors(r, c):
                  for d in DIRECTIONS:
                      nr, nc = r + d[0], c + d[1]
                      if 0 <= nr < row and 0 <= nc < col:
                          yield nr, nc

              def bfs(r, c):
                  visits[r][c] = True
                  q = collections.deque([(r, c)])

                  while q:
                      r, c = q.popleft()
                      for nr, nc in neighbors(r, c):
                          if grid[nr][nc] == WATER or visits[nr][nc]:
                              continue
                          visits[nr][nc] = True
                          q.append((nr, nc))


              if not grid or not grid[0]:
                  return 0

              row, col = len(grid), len(grid[0])
              visits = [[False for _ in range(col)] for _ in range(row)]
              island_cnt = 0
              for r in range(row):
                  for c in range(col):
                      if grid[r][c] == WATER or visits[r][c]:
                          continue

                      bfs(r, c)
                      island_cnt += 1

              return island_cnt
          ```
      * Approach2: DFS recursive,
        * Python
          ```python
          NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

          WATER = '0'
          LAND = '1'

          class Solution:
              def numIslands(self, grid: List[List[str]]) -> int:
                  def dfs(r, c):
                      if not 0 <= r < row or not 0 <= c < col:
                          return

                      if grid[r][c] == WATER or visits[r][c]:
                          return

                      visits[r][c] = True
                      for n in NEIGHBORS:
                          nr, nc = r + n[0], c + n[1]
                          dfs(nr, nc)

                  if not grid:
                      return 0

                  row, col = len(grid), len(grid[0])
                  visits = [[False for _ in range(col)] for _ in range(row)]
                  island_cnt = 0
                  for r in range(row):
                      for c in range(col):
                          if grid[r][c] == LAND and not visits[r][c]:
                              dfs(r, c)
                              island_cnt += 1

                  return island_cnt
          ```
      * Approach3: Union Find (Disjoint Set)
        * Init:
          * Union find structure with n subsets (each 1 is a subset)
        * only need to go right and go down
        * Python
          ```python
          LAND, WATER = '1', '0'
          NEIGHBORS = ((1, 0), (0, 1))

          class UnionFindSet(object):

              def __init__(self, n):
                  """
                  default parent is itself
                  """
                  self.parents = [i for i in range(n)]
                  self.ranks = [1] * (n)

              def find(self, x):
                  """
                  Find root(cluster) id
                  """
                  if x == self.parents[x]:
                      return x

                  cur = x
                  stack = []
                  while cur != self.parents[cur]:
                      stack.append(cur)
                      cur = self.parents[cur]
                  root = cur

                  """
                  Path Compression
                  """
                  while stack:
                      cur = stack.pop()
                      self.parents[cur] = root

                  return root

              def union(self, x, y):
                  r_x, r_y = self.find(x), self.find(y)

                  # x and y are in the same cluster
                  if r_x == r_y:
                      return False

                  if self.ranks[r_x] > self.ranks[r_y]:
                      self.parents[r_y] = r_x

                  elif self.ranks[r_x] < self.ranks[r_y]:
                      self.parents[r_x] = r_y

                  else:
                      self.parents[r_y] = r_x
                      self.ranks[r_x] += 1

                  return True


          class Solution:
              def numIslands(self, grid: List[List[str]]) -> int:
                  def coor_to_idx(r, c):
                      return col * r + c

                  def neighbors(r, c):
                      for d in NEIGHBORS:
                          nr, nc = r + d[0], c + d[1]
                          if 0 <= nr < row and 0 <= nc < col:
                              yield nr, nc

                  if not grid:
                      return 0

                  row, col = len(grid), len(grid[0])
                  island_cnt = 0
                  for r in range(row):
                      for c in range(col):
                          if grid[r][c] == LAND:
                              island_cnt += 1

                  ufs = UnionFindSet(row*col)
                  for r in range(row):
                      for c in range(col):
                          if grid[r][c] == WATER:
                              continue

                          idx1 = coor_to_idx(r, c)
                          for nr, nc in neighbors(r, c):
                              if grid[nr][nc] == WATER:
                                  continue
                              idx2 = coor_to_idx(nr, nc)
                              if ufs.union(idx1, idx2):
                                  island_cnt -= 1

                  return island_cnt
          ```
    * 695: Max Area of Island (M)
      * Approach1, BFS, Time:O(mn), Space:O(mn)
        * Python
          ```python
          DIRECTIONS = ((1, 0), (0, -1), (-1, 0), (0, 1))
          WATER = '0'
          LAND = '1'

          def numIslands(self, grid: List[List[str]]) -> int:
              def neighbors(r, c):
                  for d in DIRECTIONS:
                      nr, nc = r + d[0], c + d[1]
                      if 0 <= nr < row and 0 <= nc < col:
                          yield nr, nc

              def bfs(r, c):
                  visits[r][c] = True
                  q = collections.deque([(r, c)])

                  while q:
                      r, c = q.popleft()
                      for nr, nc in neighbors(r, c):
                          if grid[nr][nc] == WATER or visits[nr][nc]:
                              continue
                          visits[nr][nc] = True
                          q.append((nr, nc))


              if not grid or not grid[0]:
                  return 0

              row, col = len(grid), len(grid[0])
              visits = [[False for _ in range(col)] for _ in range(row)]
              island_cnt = 0
              for r in range(row):
                  for c in range(col):
                      if grid[r][c] == WATER or visits[r][c]:
                          continue

                      bfs(r, c)
                      island_cnt += 1

              return island_cnt
          ```
      * Approach2, DFS, Time:O(mn), Space:O(mn)
        * Python
          ```python
          NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))
          WATER = 0
          LAND = 1

          def maxAreaOfIsland(self, grid: List[List[int]]) -> int:

              def dfs(r, c):
                  if not 0 <= r < row or not 0 <= c < col:
                      return 0

                  if grid[r][c] == WATER or visits[r][c]:
                      return 0

                  visits[r][c] = True
                  area = 1
                  for n in NEIGHBORS:
                      nr, nc = r + n[0], c + n[1]
                      area += dfs(nr, nc)

                  return area

              if not grid:
                  return 0

              row, col = len(grid), len(grid[0])
              visits = [[False for _ in range(col)] for _ in range(row)]
              max_area = 0
              for r in range(row):
                  for c in range(col):
                      if grid[r][c] == LAND and not visits[r][c]:
                          max_area = max(max_area, dfs(r, c))

              return max_area
          ```
    * 694: Number of Distinct Islands (M)
      * Description:
        * Count the number of distinct islands.
        * An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.
      * How to determine the shape ??
      * Approach1: BFS, Hash by Coordinates, Time:O(mn), Space:O(mn)
        * Python
          ```python
          NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))
          WATER = 0
          LAND = 1

          def numDistinctIslands(self, grid: List[List[int]]) -> int:
              def bfs(r0, c0):
                  visits[r0][c0] = True
                  shape = [(0, 0)]
                  q = collections.deque([(r0, c0)])

                  while q:
                      r, c = q.popleft()
                      for n in NEIGHBORS:
                          nr, nc = r + n[0], c + n[1]

                          if not 0 <= nr < row or not 0 <= nc < col:
                              continue

                          if grid[nr][nc] == WATER or visits[nr][nc]:
                              continue

                          visits[nr][nc] = True
                          shape.append((nr-r0, nc-c0))
                          q.append((nr, nc))

                  return shape

                if not grid:
                    return 0

                row, col = len(grid), len(grid[0])
                visits = [[False for _ in range(col)] for _ in range(row)]
                shapes = set()
                for r in range(row):
                    for c in range(col):
                        if grid[r][c] == LAND and not visits[r][c]:
                            shapes.add(tuple(bfs(r, c)))

                return len(shapes)
          ```
      * Approach1: DFS, Hash by Coordinates, Time:O(mn), Space:O(mn)
        * Python
          ```python
          NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))
          WATER = 0
          LAND = 1

          def numDistinctIslands(self, grid: List[List[int]]) -> int:
              def dfs(r, c, r0, c0, shape):
                  if not 0 <= r < row or not 0 <= c < col:
                      return

                  if grid[r][c] == WATER or visits[r][c]:
                      return

                  visits[r][c] = True
                  shape.append((r-r0, c-c0))
                  for n in NEIGHBORS:
                      nr, nc = r + n[0], c + n[1]
                      dfs(nr, nc, r0, c0, shape)

              if not grid:
                  return 0

              row, col = len(grid), len(grid[0])
              visits = [[False for _ in range(col)] for _ in range(row)]
              shapes = set()
              for r in range(row):
                  for c in range(col):
                      if grid[r][c] == LAND and not visits[r][c]:
                          shape = []
                          dfs(r, c, r, c, shape)
                          shapes.add(tuple(shape))

              return len(shapes)
          ```
    * 463: Island Perimeter (E)
      * Approach1: BFS, Time:O(mn), Space:O(mn)
      * Approach2: Simple iteration, Time:O(mn), Space:O(1)
        * Python
          ```python
          DIRECTIONS = ((1, 0), (0, 1))
          WATER = 0
          LAND = 1

          def islandPerimeter(self, grid: List[List[int]]) -> int:
              def neighbors(r, c):
                  for d in DIRECTIONS:
                      nr, nc = r + d[0], c + d[1]
                      if 0 <= nr < row and 0 <= nc < col:
                          yield nr, nc

              row, col = len(grid), len(grid[0])
              cnt = 0
              nb_cnt = 0
              for r in range(row):
                  for c in range(col):
                      if grid[r][c] == WATER:
                          continue

                      cnt += 1
                      # only check right and down directions
                      for nr, nc in neighbors(r, c):
                          if grid[nr][nc] == LAND:
                              nb_cnt += 1

              return cnt * 4 - nb_cnt * 2
          ```
  * **Nested**
    * 339: Nested List Weight Sum (E)
      * Description
        * The weight is defined from top down.
        * n: total number of nested elements
        * d: maximum level of nesting in the input
        * example:
          * Input: [[1,1],2,[1,1]]
          * Output: 10 = 1 * 2 + 2 * 4
      * Approach1: BFS, Time: O(n), Space: O(n)
        * Python:
          ```python
          def depthSum(self, nestedList: List[NestedInteger]) -> int:
            if not nestedList:
                return 0

            q = collections.deque()

            """
            type(nestedList): list
            type(n): NestedInteger
            """
            for n in nestedList:
                q.append(n)

            sum_depth = 0
            depth = 1
            while q:
                q_len = len(q)
                for _ in range(q_len):
                    n = q.popleft()
                    if n.isInteger():
                        sum_depth += (depth * n.getInteger())
                    else:
                        for nxt_n in n.getList():
                            q.append(nxt_n)

                depth += 1

            return sum_depth
          ```
      * Approach2: DFS Recursice, Time: O(n), Space: O(n)
        * Python Solution
          ```python
          def depthSum(self, nestedList: List[NestedInteger]) -> int:
            def dfs(n, d):
                sum_depth = 0

                if n.isInteger():
                    sum_depth += (d * n.getInteger())
                else:
                    for nxt_n in n.getList():
                        sum_depth += dfs(nxt_n, d+1)

                return sum_depth

            if not nestedList:
                return 0

            """
            type(nestedList): list
            type(n): NestedInteger
            """
            sum_depth = 0
            for n in nestedList:
              sum_depth += dfs(n, 1)

            return sum_depth
          ```
      * Approach3: DFS Iteraitve, Time: O(n), Space: O(d)
        * Python
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
                    sum_depth += (d * item.getInteger())
                else:
                    for n in item.getList():
                        stack.append((n, d+1))

            return sum_depth
          ```
    * 364: Nested List Weight Sum II (M)
      * Description:
        * The weight is defined from bottom up.
        * n: total number of nested elements
        * d: maximum level of nesting in the input
        * example:
          * Input: [1,[4,[6]]]
          * Output: 17 = 1 * 3 + 4 * 2 + 6 * 1
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
        * Python
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
          max_depth = 1
          for depth in d.keys():
              max_depth = max(max_depth, depth)

          sum_depth = 0
          for depth, val in d.items():
              sum_depth += (val *(max_depth-depth+1))

          return sum_depth
          ```
    * 565: Array Nesting (M)
      * Description:
        * A zero-indexed array A of **length N** contains all integers from **0 to N-1**.
        * N is an integer within the range [1, 20,000].
        * The elements of A are all distinct.
        * Each element of A is an integer within the range [0, N-1].
        * Example:
          * Input:
            * A = [5,4,0,3,1,6,2]
          * Output:
            * One of the longest S[K]:
              * S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
      * Approach1:
        * DFS + memo, Time:O(n), Space:O(n)
          * Python
            ```python
            def arrayNesting(self, nums: List[int]) -> int:
              def get_circle_len(start):
                  cur = start
                  circle_len = 0
                  while True:
                      if visited[cur]:
                          break

                      # new node
                      visited[cur] = True
                      circle_len += 1

                      # go to next node
                      cur = nums[cur]

                  return circle_len

              """
              The elements of A are all distinct.
              Each element of A is an integer within the range [0, N-1].
              """
              if not nums:
                  return 0

              visited = [False] * len(nums)
              max_circle_len = 0

              for i in range(len(nums)):
                  if not visited[i]:
                      max_circle_len = max(max_circle_len,
                                           get_circle_len(start=i))

              return max_circle_len
            ```
  * 286: Walls and Gates (M)
    * Description:
      * -1: A wall or an obstacle.
      * 0: A gate.
      * INF: Infinity means an empty room. We use the value 2^31 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
      * Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
    * Approach1: BFS, search from **EMPTY**, Time: O((mn)^2), Space: O(mn))
      * This approach can not reuse the calculation info.
      * Time: O(**(rc)^2**)
      * Space: O(rc))
      * Python:
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
    * Approach2: BFS, search from **GATE**, Time: O((mn)), Space: O(mn))
      * This approach **can avoid recalculation**.
      * Time: O((rc))
      * Space: O(rc))
        * Queue Size
      * Python:
        ```python
        WALL = -1
        GATE = 0
        EMPTY = 2147483647
        NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

        def wallsAndGates(self, rooms: List[List[int]]) -> None:
            """
            Do not return anything, modify rooms in-place instead.
            """
            if not rooms:
                return

            row, col = len(rooms), len(rooms[0])

            q = collections.deque()
            for r in range(row):
                for c in range(col):
                    if rooms[r][c] == GATE:
                        q.append((r,c))

            d = 0
            while q:
                q_len = len(q)
                d += 1
                for _ in range(q_len):
                    r, c = q.popleft()

                    # for each neighbor
                    for n in NEIGHBORS:
                        nr, nc = r + n[0], c + n[1]

                        # check boundary
                        if not 0 <= nr < row or not 0 <= nc < col:
                            continue

                        # skip the WALL and GATE
                        if rooms[nr][nc] != EMPTY:
                            continue

                        rooms[nr][nc] = d
                        q.append((nr, nc))
          ```
  * 130: Surrounded Regions (M)
    * Description
      * A region is captured by flipping all 'O's into 'X's in that surrounded region.
    * Approach2: BFS: Time:O(rc), Space:O(rc)
      * Try to Group region
      * Python
        ```python
        NEIGHBORS = ((1, 0), (0, -1), (-1, 0), (0, 1))

        def solve(self, board: List[List[str]]) -> None:
            def _flip_surround_region(r, c):
                q = collections.deque([(r, c)])
                flip_regions = [(r, c)]
                visits[r][c] = True
                should_flip = True

                while q:
                    r, c = q.popleft()

                    for n in NEIGHBORS:
                        nr, nc = r + n[0], c + n[1]

                        if not 0 <= nr < row or not 0 <= nc < col:
                            """
                            do not break here since we try to extend the max regino as possible
                            """
                            should_flip = False
                            continue

                        if board[nr][nc] == 'X' or visits[nr][nc]:
                            continue

                        # group region
                        if board[nr][nc] == 'O':
                            visits[nr][nc] = True
                            q.append((nr, nc))
                            if should_flip:
                              flip_regions.append((nr, nc))


                if should_flip:
                  while flip_regions:
                      r, c = flip_regions.pop()
                      board[r][c] = 'X'


            if not board:
                return

            row, col = len(board), len(board[0])
            visits = [[False for _ in range(col)] for _ in range(row)]

            for r in range(row):
                for c in range(col):
                    if board[r][c] == 'O' and not visits[r][c]:
                        _flip_surround_region(r, c)
        ```
  * 127: **Word Ladder** (M)
    * Description:
        * n: length of wordList
        * k: length of beginWord
        * Example:
          * Input:
            * beginWord = "hit",
            * endWord = "cog",
            * wordList = ["hot","dot","dog","lot","log","cog"]
          * Output:
            * As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog"
    * Approach1: BFS, Time:O(nk), Space:O((n * k)*k)
      * Time: O(n*k)
        * Build transformation dict cost: O(n*k)
        * Find the target word in the transformation dict cost: O(n*k)
      * Space: O(n*k^2)
        * Transformatino dict cost: O((n * k)*k)
          * Each word has k transformations, each transformation cost k
        * Max queue size: O((n * k))
      * Python
        ```python
        def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
          if not wordList:
              return 0

          if beginWord == endWord:
              return 1

          """
          Generate word dictionary,
          hit -> *it, h*t, hi*
          """
          # generate the word dict
          w_dict = collections.defaultdict(list)
          for w in wordList:
              for i in range(len(w)):
                  pattern = f'{w[:i]}*{w[i+1:]}'
                  w_dict[pattern].append(w)


          transformd_cnt = 1
          memo = dict()
          q = collections.deque([beginWord])

          while q:
              q_len = len(q)
              transformd_cnt += 1
              for _ in range(q_len):
                  w = q.popleft()

                  for i in range(len(w)):
                      pattern = f'{w[:i]}*{w[i+1:]}'

                      if pattern not in w_dict:
                          continue

                      for transform in w_dict[pattern]:
                          if transform == endWord:
                              return transformd_cnt

                          if transform in memo:
                              continue

                          memo[transform] = True
                          q.append(transform)

                      w_dict.pop(pattern)

          return 0
        ```
    * Approach2: Bidirectional BFS
  * 126: Word Ladder II (H)
  * 079: **Word Search** (M)
    * Description:
     * L is the word length
     * The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those **horizontally** or **vertically** neighboring. The same letter cell **can not be used more than once**.
     * Note
       * Notice the boundary handling, for example: board: [["a"]],  word:"a"
       * Need to clean the visited infomation for each traverse.
    * Approach1: DFS Recursive: Time:O(mn*4^L), Space:O(mn)
       * Time:
         * The first character takes O(mn)
         * Remain characters take O(4^L)
       * Python
         ```python
          DIRECTIONS = ((1, 0), (0, -1), (-1, 0), (0, 1))

          def exist(self, board: List[List[str]], word: str) -> bool:
              def neighbors(r, c):
                  for d in DIRECTIONS:
                      nr, nc = r + d[0], c + d[1]
                      if 0 <= nr < row and 0 <= nc < col:
                          yield nr, nc

              def dfs(idx, r, c):
                  if board[r][c] != word[idx]:
                      return False

                  """
                  note1: can not use idx == len(word)
                  for example: board: [["a"]],  word:"a"
                  """
                  if idx == len(word)-1:
                      return True

                  is_found = False
                  visited[r][c] = True
                  for nr, nc in neighbors(r, c):
                      if not visited[nr][nc]:
                          is_found = dfs(idx+1, nr, nc)
                          if is_found:
                              break
                  """
                  note2: backtrak, restore the vistied information
                  """
                  visited[r][c] = False
                  return is_found

              if not board or not board[0]:
                  return False

              if not word:
                  return True

              row, col = len(board), len(board[0])
              visited = [[False for _ in range(col)] for _ in range(row)]
              is_found = False
              for r in range(row):
                  for c in range(col):
                      is_found = dfs(0, r, c)
                      if is_found:
                          break
                  else:
                      continue
                  break

              return is_found
         ```
    * Approach2: DFS Iterative: Time:O(mn*4^L), Space:O(mn)
       * Use another stack to track unsed info
       * Python
         ```python
         DIRECTIONS = ((1, 0), (0, -1), (-1, 0), (0, 1))
        def exist(self, board: List[List[str]], word: str) -> bool:
            def neighbors(r, c):
                for d in DIRECTIONS:
                    nr, nc = r + d[0], c + d[1]
                    if 0 <= nr < row and 0 <= nc < col:
                        yield nr, nc

            def dfs(idx, r, c):
                is_found = False
                s = [(idx, r, c, False)]

                while s:
                    idx, r, c, backtrack = s.pop()
                    if backtrack:
                        visited[r][c] = False
                        continue

                    if board[r][c] != word[idx]:
                        continue

                    """
                    note1: can not use idx == len(word)
                    for example: board: [["a"]],  word:"a"
                    """
                    if idx == len(word)-1:
                        is_found = True
                        break

                    visited[r][c] = True
                    """
                    note2: backtrack the visited info
                    """
                    s.append((-1, r, c, True))
                    for nr, nc in neighbors(r, c):
                        if not visited[nr][nc]:
                            s.append((idx+1, nr, nc, False))

                return is_found

            if not board or not board[0]:
                return False

            if not word:
                return True

            row, col = len(board), len(board[0])
            visited = [[False for _ in range(col)] for _ in range(row)]
            is_found = False
            for r in range(row):
                for c in range(col):
                    is_found = dfs(0, r, c)
                    if is_found:
                        break
                else:
                    continue
                break

            return is_found
         ```
  * 329: **Longest Increasing Path** in a Matrix (H)
    * Description:
      * Given an integer matrix, find the length of the longest increasing path.
      * From each cell, you can either move to four directions: left, right, up or down.
    * Note:
      * Do not need to keep visited due to the increasing property.
    * Approach1: DFS + memo, Time:O((mn)), Space:O(mn)
      * Python
        ```python
        DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
        def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
            def neighbors(r, c):
                for d in DIRECTIONS:
                    nr, nc = r + d[0], c + d[1]
                    if 0 <= nr < row and 0 <= nc < col:
                        yield nr, nc

            def dfs(r, c):
                if memo[r][c]:
                    return memo[r][c]

                lip = 1
                for nr, nc in neighbors(r, c):
                    if matrix[r][c] < matrix[nr][nc]:
                        lip = max(lip, 1+dfs(nr, nc))
                memo[r][c] = lip

                return memo[r][c]

            if not matrix or not matrix[0]:
                return 0

            row, col = len(matrix), len(matrix[0])
            memo = [[None for _ in range(col)] for _ in range(row)]
            lip = 0
            for r in range(row):
                for c in range(col):
                    lip = max(lip, dfs(r, c))

            return lip
        ```
    * Approach2: Topological Sort, Time:O((mn)), Space:O(mn)
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
        DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
        def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
            def neighbors(r, c):
                for d in DIRECTIONS:
                    nr, nc = r + d[0], c + d[1]
                    if 0 <= nr < row and 0 <= nc < col:
                        yield nr, nc

            if not matrix or not matrix[0]:
                return 0

            row, col = len(matrix), len(matrix[0])
            indegree = [[0 for _ in range(col)] for _ in range(row)]

            # create the graph
            for r in range(row):
                for c in range(col):
                    for nr, nc in neighbors(r, c):
                        if matrix[nr][nc] > matrix[r][c]:
                            indegree[nr][nc] += 1

            # topoligical sorting
            q = collections.deque()
            for r in range(row):
                for c in range(col):
                    if indegree[r][c] == 0:
                        q.append((r,c))

            lip = 0
            while q:
                lip += 1
                q_len = len(q)
                for _ in range(q_len):
                    r, c = q.popleft()
                    for nr, nc in neighbors(r, c):
                        if matrix[nr][nc] > matrix[r][c]:
                            indegree[nr][nc] -= 1
                            if indegree[nr][nc] == 0:
                                q.append((nr,nc))

            return lip
        ```
  * 051: N-Queens (H)
  * 052: N-Queens II (H)
  * 317: Shortest Distance from All Buildings
### Union Find (Disjoint-set)
  * Tips:
    * What is Union Find Set
      * Ref:
        * https://www.youtube.com/watch?v=VJnUwsE4fWA
      * Find(x): find the root/cluster-id of x
        * amortized O(1)
        * **path compression for ancestors**
      * union(x, y): merge two clusters
        * amortized O(1)
        * **Merge low rank tree into high rank one.**
        * If two sub-tree have the same rank, break tie arbitrarily and increase the merged tree's rank by
      * check whether two elements are in the same cluster
        * Optimization: amortized O(1)
          * Path compression: make tree flat
          * Union by rank: merge low rank tree to high rank tree
        * Without Optimization costs O(n)
      * Space:
        * O(n)
     * Implementation:
       * Python
        ```python
        class UnionFindSet(object):
          def __init__(self, n):
              """
              default parent is itself
              """
              self.parents = [i for i in range(n+1)]
              self.ranks = [1] * (n+1)

          def find(self, x):
              """
              Find root(cluster) id
              Time:O(1)
              Space:O(n)
              """
              if x == self.parents[x]:
                  return x

              cur = x
              stack = []
              while cur != self.parents[cur]:
                  stack.append(cur)
                  cur = self.parents[cur]
              root = cur

              """
              Path Compression
              Update root of ancestors
              """
              while stack:
                  cur = stack.pop()
                  self.parents[cur] = root

              return root

          def union(self, x, y):
              """
              Time: O(1)
              """
              r_x, r_y = self.find(x), self.find(y)

              # x and y are in the same cluster
              if r_x == r_y:
                  return False

              if self.ranks[r_x] > self.ranks[r_y]:
                  self.parents[r_y] = r_x

              elif self.ranks[r_x] < self.ranks[r_y]:
                  self.parents[r_x] = r_y

              else:
                  self.parents[r_y] = r_x
                  self.ranks[r_x] += 1

              return True
        ```
  * Kruskal's algorithm:
    * 1135: Connecting Cities With Minimum Cost (M)
      * Approach1: Kruskal, Time:O(ElogE), Space:O(V)
        * Time:
          * sorting by cost: ElogE
          * union E
        * Space:
          * O(V)
        * Python
          ```python
          class UnionFindSet(object):

            def __init__(self, n):
                """
                index 0 is reserved
                """
                self.parents = [i for i in range(n+1)]
                self.ranks = [1] * (n+1)
                self.cnt = n

            def find(self, x):
                """
                Find root id
                """
                if x == self.parents[x]:
                    return x

                cur = x
                stack = []
                while cur != self.parents[cur]:
                    stack.append(cur)
                    cur = self.parents[cur]
                root = cur

                """
                Path Compression
                """
                while stack:
                    cur = stack.pop()
                    self.parents[cur] = root

                return root

            def union(self, x, y):
                r_x, r_y = self.find(x), self.find(y)

                # in the same cluster
                if r_x == r_y:
                    return False

                if self.ranks[r_x] > self.ranks[r_y]:
                    self.parents[r_y] = r_x

                elif self.ranks[r_x] < self.ranks[r_y]:
                    self.parents[r_x] = r_y

                else:
                    self.parents[r_y] = r_x
                    self.ranks[r_x] += 1

                self.cnt -= 1
                return True


            def minimumCost(self, N: int, connections: List[List[int]]) -> int:
                connections.sort(key=lambda x:x[2])
                ufs = UnionFindSet(N)
                cost = 0

                for u, v, c in connections:
                    if ufs.find(u) == ufs.find(v):
                        continue

                    ufs.union(u, v)
                    cost += c

                return cost if ufs.cnt == 1 else -1
            ```
  * 684: Redundant Connection (M)
    * Description:
      * In this problem, a tree is an undirected graph that is connected and **has no cycles**.
      * The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. **The added edge has two different vertices chosen from 1 to N**, **and was not an edge that already existed**.
      * Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array.
    * Approach1: DFS, Time:O(|E|^2), Space:O(|E|)
      * Python
        ```python
        def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
          def dfs(src, target):
              visited = {src: True}
              stack = [src]
              is_found = False
              while stack:
                  cur = stack.pop()
                  if cur == target:
                      is_found = True
                      break

                  for nb in graph[cur]:
                      if nb in visited:
                          continue

                      visited[nb] = True
                      stack.append(nb)

              return is_found

          graph = collections.defaultdict(set)
          for u, v in edges:
              """
              check if u and v have been connected
              """
              if dfs(src=u, target=v):
                  res = [u, v]
              """
              add edges (u, v) and (v, u)
              """
              graph[u].add(v)
              graph[v].add(u)
          return res
         ```
    * Approach2: Union Find, Time:O(|E|), Space:O(|E|)
      * Python
        ```python
        class UnionFindSet(object):

          def __init__(self, n):
              """
              default parent is itself
              """
              self.parents = [i for i in range(n+1)]
              self.ranks = [1] * (n+1)

          def find(self, x):
              """
              Find root(cluster) id
              Time:O(1)
              Space:O(n)
              """
              if x == self.parents[x]:
                  return x

              cur = x
              stack = []
              while cur != self.parents[cur]:
                  stack.append(cur)
                  cur = self.parents[cur]
              root = cur

              """
              Path Compression
              Update root of ancestors
              """
              while stack:
                  cur = stack.pop()
                  self.parents[cur] = root

              return root

          def union(self, x, y):
              """
              Time: O(1)
              """
              r_x, r_y = self.find(x), self.find(y)

              # x and y are in the same cluster
              if r_x == r_y:
                  return False

              if self.ranks[r_x] > self.ranks[r_y]:
                  self.parents[r_y] = r_x

              elif self.ranks[r_x] < self.ranks[r_y]:
                  self.parents[r_x] = r_y

              else:
                  self.parents[r_y] = r_x
                  self.ranks[r_x] += 1

              return True


          def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
              ufs = UnionFindSet(len(edges))
              res = None
              for edge in edges:
                """
                check if node u and node v are in the same cluster
                """
                if ufs.find(edge[0]) == ufs.find(edge[1]):
                    res = edge
                    break
                ufs.union(*edge)

              return res
        ```
  * 685: Redundant Connection II (H)
  * 261: Graph Valid Tree (M)
    * Approach1: Union Find, Time:O(E), Space:O(V)
      * Python
        ```python
        class UnionFindSet(object):

          def __init__(self, n):
              self.parents = [i for i in range(n)]
              self.ranks = [1] * (n)
              self.cnt = n

          def find(self, x):
              """
              Find root id
              """
              if x == self.parents[x]:
                  return x

              cur = x
              stack = []
              while cur != self.parents[cur]:
                  stack.append(cur)
                  cur = self.parents[cur]
              root = cur

              """
              Path Compression
              """
              while stack:
                  cur = stack.pop()
                  self.parents[cur] = root

              return root

          def union(self, x, y):
              r_x, r_y = self.find(x), self.find(y)

              # in the same cluster
              if r_x == r_y:
                  return False

              if self.ranks[r_x] > self.ranks[r_y]:
                  self.parents[r_y] = r_x

              elif self.ranks[r_x] < self.ranks[r_y]:
                  self.parents[r_x] = r_y

              else:
                  self.parents[r_y] = r_x
                  self.ranks[r_x] += 1

              self.cnt -= 1

              return True

          def validTree(self, n: int, edges: List[List[int]]) -> bool:
              ufs = UnionFindSet(n)
              valid_tree = True
              for u, v in edges:
                  if ufs.find(u) == ufs.find(v):
                      valid_tree = False
                      break

                  ufs.union(u, v)

              return valid_tree and ufs.cnt == 1
        ```
  * 399: Evaluate Division (M)
  * 547: Friend Circles (M)
    * Description:
      * There are **N students** in a class
      * Their friendship is **transitive** in nature. For example, if A is a direct friend of B, and B is a direct friend of C, then A is an indirect friend of C.
      * And we defined a friend circle is a group of **students who are direct or indirect friends.**
      * Given a N*N matrix M representing the friend relationship between students in the class. If M[i][j] = 1, then the ith and jth students are direct friends with each other
    * Note
      * Our problem reduces to the problem of finding the number of connected components in an undirected graph.
      * Matrix is a adjacency list graph
    * Approach1: DFS, Time:O(n), Space:O(n)
      * Time complexity is bounded by visited array
      * Python
        ```python
        def findCircleNum(self, M: List[List[int]]) -> int:
          def dfs(s):
              for f in range(n):
                  if not visited[f] and M[s][f] == 1:
                      visited[f] = True
                      dfs(f)

          n = len(M)
          visited = [False] * n
          circle_cnt = 0
          for s in range(n):
              if visited[s]:
                  continue
              dfs(s)
              circle_cnt += 1

          return circle_cnt
        ```
    * Approach2: BFS, Time:O(n), Space:O(n)
      * * Time complexity is bounded by visited array
      * Python
        ```python
        def findCircleNum(self, M: List[List[int]]) -> int:
          def bfs(s):
              q = collections.deque([s])
              visited[s] = True
              while q:
                  s = q.popleft()
                  for f in range(n):
                      if not visited[f] and M[s][f] == 1:
                          visited[f] = True
                          q.append(f)

        n = len(M)
        visited = [False] * n
        circle_cnt = 0
        for s in range(n):
            if visited[s]:
                continue
            bfs(s)
            circle_cnt += 1

        return circle_cnt

        ```
    * Approach3: Union Find, Time:O(n^2), Space:O(n)
      * Python
        ```python
        class UnionFindSet(object):

          def __init__(self, n):
              self.parents = [i for i in range(n)]
              self.ranks = [1] * n
              self.cnt = n

          def find(self, x):
              """
              Find root id
              """
              if x == self.parents[x]:
                  return x

              cur = x
              stack = []
              while cur != self.parents[cur]:
                  stack.append(cur)
                  cur = self.parents[cur]
              root = cur

              """
              Path Compression
              """
              while stack:
                  cur = stack.pop()
                  self.parents[cur] = root

              return root

          def union(self, x, y):
              r_x, r_y = self.find(x), self.find(y)

              # in the same cluster
              if r_x == r_y:
                  return False

              if self.ranks[r_x] > self.ranks[r_y]:
                  self.parents[r_y] = r_x

              elif self.ranks[r_x] < self.ranks[r_y]:
                  self.parents[r_x] = r_y

              else:
                  self.parents[r_y] = r_x
                  self.ranks[r_x] += 1

              self.cnt -= 1

              return True

        def findCircleNum(self, M: List[List[int]]) -> int:
            row = col = len(M)
            ufs = UnionFindSet(len(M))
            # student and friend
            for s in range(row):
                for f in range(col):
                    if s != f and M[s][f] == 1:
                        ufs.union(s, f)

            return ufs.cnt
        ```
  * 737: Sentence Similarity II (M)
    * Description:
      * transitive
        * For example, if "great" and "good" are similar, and "fine" and "good" are similar, then "great" and "fine" are similar
      * symmetric
        * For example, "great" and "fine" being similar is the same as "fine" and "great" being similar.
      * A word is always similar with itself
    * n is the number of words, p is the number of pairs
    * Approach1: DFS, Time:O(np), Space:O(p)
      * Python
        ```python
        def areSentencesSimilarTwo(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:

          def dfs(src_w, targt_w):
              if src_w == targt_w:
                  return True

              visited[src_w] = True
              if src_w not in w_graph :
                  return False

              is_found = False
              for transform in w_graph[src_w]:
                  if transform in visited:
                      continue

                  if dfs(transform, targt_w):
                      is_found = True
                      break

              return is_found


          if len(words1) != len(words2):
              return False

          w_graph = collections.defaultdict(list)
          for u, v in pairs:
              w_graph[u].append(v)
              w_graph[v].append(u)

          is_similarity = True
          for src, target in zip(words1, words2):
              visited = dict()
              if not dfs(src, target):
                  is_similarity = False
                  break

          return is_similarity
        ```
    * Approach2: BFS, Time:O(np), Space:O(p)
    * Approach3: Union Find, Time:O(n+p), Space:O(p)
      * Time:
        * Create the word to index dict: O(p)
        * Init the ufs: O(p)
        * Check if w1, w2 are in the same cluster: O(n)
      * Space:
        * word to index dict: O(p)
        * Union: O(p)
      * Python
        ```python
        class UnionFindSet(object):

          def __init__(self, n):
              self.parents = [i for i in range(n)]
              self.ranks = [1] * (n)

          def find(self, x):
              """
              Find root id
              """
              if x == self.parents[x]:
                  return x

              cur = x
              stack = []
              while cur != self.parents[cur]:
                  stack.append(cur)
                  cur = self.parents[cur]
              root = cur

              """
              Path Compression
              """
              while stack:
                  cur = stack.pop()
                  self.parents[cur] = root

              return root

          def union(self, x, y):
              r_x, r_y = self.find(x), self.find(y)

              # in the same cluster
              if r_x == r_y:
                  return False

              if self.ranks[r_x] > self.ranks[r_y]:
                  self.parents[r_y] = r_x

              elif self.ranks[r_x] < self.ranks[r_y]:
                  self.parents[r_x] = r_y

              else:
                  self.parents[r_y] = r_x
                  self.ranks[r_x] += 1

              return True

          def areSentencesSimilarTwo(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:

              if len(words1) != len(words2):
                  return False

              """
              Each pair has two nodes
              """
              ufs = UnionFindSet(len(pairs) * 2)
              # map a word to an index
              d = dict()
              i = 0
              for w1, w2 in pairs:
                  """
                  index the words
                  """
                  for w in (w1, w2):
                      if w not in d:
                          d[w] = i
                          i += 1
                  ufs.union(d[w1], d[w2])

              is_similarity = True
              for w1, w2 in zip(words1, words2):
                  if w1 == w2:
                      continue

                  if w1 not in d or w2 not in d or ufs.find(d[w1]) != ufs.find(d[w2]):
                      is_similarity = False
                      break

              return is_similarity
          ```
  * 200: Number of Islands (M)
    * See BFS & DFS
  * 305: Number of Islands II (H)
  * 839: Similar String Groups
  * 959: Regions Cut By Slashes
  * 323: Number of Connected Components in an Undirected Graph (M)
  * 721: Accounts Merge (M)
### Backtracking
  * 294: Flip Game II (M)
      * Approach1: backtracking: Time: O(n!!), Space: O(n*2)
        * **Double factorial**: (n-1) * (n-3) * (n-5) * ... 1=
         * python
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
  * 022: Generate Parentheses (M)
     * Description:
       * Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
     * Approach1-1: Brute Force, Recursive, Time:O(n*2^n)
       * f(n) = f(n-1) + '('  or f(n-1) + ')'
       * Time Complexity:
         * total n^(2n) combination
       * Space Complexity:
       * Python:
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
     * Approach1-2: Brute Force, Iterative: Time:O(n*2^n)
       * Python
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
       * Python:
        ```python
        def generateParenthesis(self, n: int) -> List[str]:
          def backtrack(cur, left, right):
              if left == right == n:
                  parentheses.append(''.join(cur))
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
          parentheses = []
          backtrack(cur, left=0, right=0)

          return parentheses
        ```
     * Approach2-2: Control the left and right, Iterative:
       * Python:
          ```python
          def generateParenthesis(self, n: int) -> List[str]:
            parentheses = []
            stack = []
            # s, left ,right
            stack.append(('(', 1, 0))
            while stack:
                s, left, right = stack.pop()
                if left == right == n:
                    parentheses.append(s)
                    continue

                if left < n:
                    stack.append((s+'(', left+1, right))

                if right < left:
                    stack.append((s+')', left, right+1))

            return parentheses
          ```
     * Approach3: DP
       * Ref:
         * https://leetcode.com/problems/generate-parentheses/discuss/10127/An-iterative-method.
        ```txt
        Lets f(n) is a function which is set of valid string with n (matched) parantheses.
        f(0) = ""
        f(1) = (f(0))f(0)
        f(2) = (f(0))f(1) + (f(1))f(0)
        f(3) = (f(0))f(2) + (f(1))f(1) + (f(2))f(0)
        f(n) = ( f(0) ) f(n-1) + ( f(1) ) f(n-2) + ( f(2) ) f(n-3) + ...... + ( f(n-1) ) f(0)
        ```
      * Python
        ```python
        def generateParenthesis(self, n: int) -> List[str]:
          """
          f(0) = ""
          f(1) = (f(0))f(0)
          f(2) = (f(0))f(1) + (f(1))f(0)
          f(3) = (f(0))f(2) + (f(1))f(1) + (f(2))f(0)
          f(n) = ( f(0) ) f(n-1) + ( f(1) ) f(n-2) + ( f(2) ) f(n-3) + ...... + ( f(n-1) ) f(0)
          """
          parentheses = [[] for i in range(n+1)]
          parentheses[0].append("")

          for i in range(1, n+1):
              for left, right in zip(range(i), reversed(range(i))):
                  p_left = parentheses[left]
                  p_right = parentheses[right]
                  """
                  p_left and p_right are lists
                  """
                  for l in p_left:
                      for r in p_right:
                          parentheses[i].append(f'({l}){r}')

          return parentheses[-1]
          ```
  * **Subset** (set)
    * 078: Subsets (M)
      * Description:
        * Given a set of **distinct integers**, nums, return all possible subsets (the power set).
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
                    # make a copy
                    subs.append(sub[:])
                    # append n to the original sub
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
            def subsets(self, nums: List[int]) -> List[List[int]]:
              if not nums:
                  return []

              subs = [[] for _ in range(2**len(nums))]

              for sub_idx, sub in enumerate(subs):
                  for n_idx, n in enumerate(nums):
                      if sub_idx & (1 << n_idx):
                          sub.append(n)

              return subs
            ```
    * 090: Subsets II (M)
      * Given a collection of integers that might **contain duplicates**, nums, return all possible subsets (the power set).
      * Approach1: backtracking
        * Python
          ```python
          def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
            def backtrack(start, cur):
                subs.append(cur[:])

                if start == n:
                    return

                for i in range(start, n):
                    """
                    skip duplicate in the same position
                    for example: [1,1,2]
                                  ^ ^
                    """
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

                """
                same idea like backtracking
                we can not insert duplicate val in the same position
                """
                if i > 0 and nums[i] == nums[i-1]:
                    # the source is previous cur
                    cur = [sub + [n] for sub in cur]
                else:
                    # the source is subs
                    cur = [sub + [n] for sub in subs]

                subs.extend(cur)

            return subs
          ```
  * **Combinations**
    * 077: Combinations (M)
      * Description
        * Given two integers n and k, return all possible **combinations of k numbers out of 1 ... n**.
        * Example:
          ```txt
          Input: n = 4, k =2
          Output:
               [
                [1,2],
                [1,3],
                [1,4],
                [2,3],
                [2,4],
                [3,4]
               ]
          ```
      * Tracking Example:
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
      * Approach1: backtracking Time: O(k * n!/(n!*(n-k))!), Space: O(n!/(n!*(n-k)!)
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
                      combs.append(cur[:])
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

              combs = []
              cur = []
              backtrack(1, cur)
              return combs
            ```
      * Approach2: Iterative
        * The same idea like recursive approach, use stack to control backtrack
        * Python
          ```python
          def combine(self, n: int, k: int) -> List[List[int]]:
            if n < k or k < 1:
                return []

            combs = []
            cur = []
            start = 1
            while True:

                if len(cur) == k:
                    combs.append(cur[:])

                """
                k - l > n - start + 1 means that l will not satisfy k in the future
                in fact, (k - l) > (n - start + 1)  can cover start > n when (l-k) = -1
                The backtrack condition is the same as recursive approach1:
                1. len(cur) == k
                2. can not satisfity in the future: (k - l) > (n - start + 1)
                3. out of boundary
                """
                if len(cur) == k or k - len(cur) > n - start + 1 or start > n:
                    if not cur:
                        break

                    start = cur.pop() + 1

                else:
                    cur.append(start)
                    start += 1

            return combs
          ```
    * 039: Combination Sum (M)
      * Description:
        * Given a set of candidate numbers (candidates) (**without duplicates**) and a target number (target), **find all unique combinations** in candidates where the candidate numbers sums to target.
        * The same repeated number may be chosen from candidates unlimited number of times.
        * example:
          ```txt
          Input: candidates = [2,3,5], target = 8
          A solution set is:
              [
                [2,2,2,2],
                [2,3,3],
                [3,5]
              ]
          ```
      * Time:
        * https://leetcode.com/problems/combination-sum/discuss/16634/If-asked-to-discuss-the-time-complexity-of-your-solution-what-would-you-say
      * Approach1: backtracking:
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
      * Approach1: backtracking:
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
                  n = nums[i]
                  next_perms = []

                  for perm in perms:
                      for b in range(len(perm)+1):
                          next_perms.append(perm[:b] + [n] + perm[b:])

                  perms = next_perms

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
          def permuteUnique(self, nums: List[int]) -> List[List[int]]:
            def backtrack(counter, cur):
                if len(cur) == len(nums):
                    perms.append(cur[:])
                    return

                # don't pick duplicates in the same position
                for c, cnt in counter.items():
                    if cnt == 0:
                        continue

                    cur.append(c)
                    counter[c] -= 1
                    backtrack(counter, cur)
                    cur.pop()
                    counter[c] += 1

            if not nums:
                return []

            perms = []
            cur = []
            counter = collections.Counter(nums)
            backtrack(counter, cur)
            return perms
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
                next_perms = []
                for perm in perms:
                    for b in range(len(perm)+1):
                        next_perms.append(perm[:b] + [n] + perm[b:])
                        # To handle duplication
                        # just avoid inserting a number AFTER any of its duplicates.
                        if b < len(perm) and perm[b] == n:
                            break

                perms = next_perms
            return perms
          ```
    * 267: Palindrome Permutation II (M)
       * Description:
         * Given a string s, return all the **palindromic permutations** (without duplicates) of it. Return an empty list if no palindromic permutation could be form.
       * Approach1: backtrack, Time:O(n!), Space:O(n)
         * Python
          ```python
          def generatePalindromes(self, s: str) -> List[str]:
            def get_odd_cnt_chr(counter):
                odd_cnt = 0
                odd_chr = None
                for c, cnt in counter.items():
                    if cnt % 2:
                        odd_cnt += 1
                        odd_chr = c

                return odd_cnt, odd_chr

            def backtrack(counter, cur):
                if len(cur) == len(s):
                    perms.append("".join(cur))
                    return

                for c, cnt in counter.items():
                    if cnt < 2:
                        continue

                    cur.append(c)
                    cur.appendleft(c)
                    counter[c] -= 2
                    backtrack(counter, cur)
                    counter[c] += 2
                    cur.pop()
                    cur.popleft()


            if not s:
                return []

            counter = collections.Counter(s)
            odd_cnt, odd_chr = get_odd_cnt_chr(counter)

            # can not form valid palindrmic permutation
            if odd_cnt > 1:
                return []

            cur = collections.deque()
            if odd_cnt == 1:
                cur.append(odd_chr)

            perms = []
            backtrack(counter, cur)
            return perms
          ```
    * 784: Letter Case Permutation (E)
      * Approach1: backtracking, Time:O(n^2n), Space: O(n^2n)
        * Python
          ```python
          def letterCasePermutation(self, S: str) -> List[str]:
            def backtrack(i, cur):
                if i == len(S):
                    perms.append("".join(cur))
                    return

                if cur[i].isdigit():
                    backtrack(i+1, cur)
                else:
                    cur[i] = cur[i].upper()
                    backtrack(i+1, cur)
                    cur[i] = cur[i].lower()
                    backtrack(i+1, cur)

            if not S:
                return []

            perms = []
            cur = list(S)
            backtrack(0, cur)
            return perms
          ```
      * Approach2: iterative, Time:O(n^2n), Space: O(n^2n)
        * Python
          ```python
          def letterCasePermutation(self, S: str) -> List[str]:
            if not S:
                return []

            perms = [[]]

            for c in S:
                if c.isdigit():
                    for perm in perms:
                        perm.append(c)
                else:
                    next_perms = []
                    for lower_perm in perms:
                        # make a copy
                        upper_perm = lower_perm[:]
                        lower_perm.append(c.lower())
                        upper_perm.append(c.upper())

                        next_perms.append(lower_perm)
                        next_perms.append(upper_perm)

                    perms = next_perms

            return [''.join(perm) for perm in perms]
          ```
    * 060: Permutation Sequence (M)
      * Description:
        * Given n and k, return the kth permutation sequence.
        * example:
          * n = 3, k = 3
          ```txt
          "123"
          "132"
          "213" <--- k = 3, retrun "213"
          "231"
          "312"
          "321"
          ```
       * Ref:
         *  https://leetcode.com/problems/permutation-sequence/discuss/22507/%22Explain-like-I'm-five%22-Java-Solution-in-O(n)
      * Approach1: Time:O(n^2), Space:O(n)
        * Python
          ```python
          def getPermutation(self, n: int, k: int) -> str:
          """
          if n = 4
          cnt of permutation is
          factorial(4) = 24
          """
          permute_cnt = math.factorial(n)
          nums = [str(i) for i in range(1, n+1)]
          kth_permute = []
          k = k - 1

          # n to 1
          for first_element_cnt in range(n, 0, -1):
              permute_cnt //= first_element_cnt
              idx = k // permute_cnt

              kth_permute.append(nums[idx])
              # cost O(n), how to enhance ?
              nums.pop(idx)

              k = k % permute_cnt

          return "".join(kth_permute)
          ```
    * 031: Next Permutation (M)
  * Others:
    * 320. Generalized Abbreviation (M)
    * 291: Word Pattern II (H)
### Graph
  * **Eulerian trail**
    * A finite graph that **visits every edge** exactly once.
    * 332: Reconstruct Itinerary (M)
      * Description:
        * Given a list of airline tickets represented by pairs of **departure and arrival airports [from, to]**, reconstruct the itinerary in order.
        * All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.
        * You may assume all tickets form at least one valid itinerary.
        * Note:
          * **If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order** when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
          * You may assume all tickets form at least one valid itinerary.
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
        * Python:
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
        * Python
          ```python
          def findItinerary(self, tickets: List[List[str]]) -> List[str]:
            g_outdegree = collections.defaultdict(list)
            # create the graph
            for departure, arrival in tickets:
                g_outdegree[departure].append(arrival)

            # gnerate the min-heap for each arrival list
            for arrivals in g_outdegree.values():
                heapq.heapify(arrivals)

            itinerary = collections.deque()

            stack =["JFK"]
            while stack:
                departure = stack[-1]
                arrivials = g_outdegree[departure]
                if arrivials:
                    stack.append(heapq.heappop(arrivials))
                else:
                    stack.pop()
                    itinerary.appendleft(departure)

            return itinerary
          ```
  * **Eulerian circuit**
    * An **Eulerian trail** that **starts and ends on the same vertex**.
  * **Hamilton path**
    * A finite graph that **visits every vertex** exactly once.
  * **Hamilton cycle**
    * An Hamilton path that **starts and ends on the same vertex**.
  * **Minimum Spanning Tree**
    * A minimum spanning tree (MST) or minimum weight spanning tree is **a subset of the edges of a connected, edge-weighted undirected graph that connects all the vertices together, without any cycles and with the minimum possible total edge weight**.
  * **Shortest Path**
  * Topological Sort
    * Ref:
      * https://www.youtube.com/watch?v=ddTC4Zovtbc
      * https://leetcode.com/problems/course-schedule-ii/solution/
    * 207: Course Schedule (M)
      * Description:
        * There are a total of n courses you have to take, labeled from 0 to n-1.
        * Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
          * [course, prequisite]
      * Approach1: Peeling Onion, Time: O(V+E), Space: O(V+E)
        * Time: O(V+E)
          * Build Outdegree List Graph and Indegree Array
            * O(E)
          * Traverse the Nodes and Edges: O(V+E)
            * Traverse the Nodes:
                * Each node would be traverse at most once.
                * Operation for push node to queue and pop node from queue.
            * Traverse the Edges.
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
        * Python:
          ```python
          def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
            g_outdegree = collections.defaultdict(list)
            g_indegree = [0] * numCourses

            for course, preq in prerequisites:
                g_outdegree[preq].append(course)
                g_indegree[course] += 1

            """
            Starting from the courses having 0 prequisite (indegree is 0)
            """
            q = collections.deque()
            for course, indegree in enumerate(g_indegree):
                if indegree == 0:
                    q.append(course)

            scheduled_courses = 0
            while q:
                cur = q.popleft()
                schedule_cnt += 1

                for nxt in g_outdegree[cur]:
                    g_indegree[nxt] -= 1
                    if g_indegree[nxt] == 0:
                        q.append(nxt)

            return scheduled_courses == numCourses
          ```
      * Approach2: DFS:
        * Algorithm:
          * For each of the nodes in our graph, we will run a depth first search in case that node was not already visited in some other node's DFS traversal.
          * Suppose we are executing the depth first search for a node N. We will recursively traverse all of the neighbors of node N which have not been processed before.
          * Once the processing of all the neighbors is done, we will add the node N to the stack. We are making use of a stack to simulate the ordering we need. When we add the node N to the stack, all the nodes that require the node N as a prerequisites (among others) will already be in the stack.
        * Python
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
      * Approach1: Peeling Onion
        * Python
          ```python
          def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

          g_outdegree = collections.defaultdict(list)
          g_indegree = [0] * numCourses

          for course, preq in prerequisites:
              g_outdegree[preq].append(course)
              g_indegree[course] += 1

          """
          Traverse from the courses having 0 prequisite (indegree is 0)
          """
          q = collections.deque()
          for course, indegree in enumerate(g_indegree):
              if indegree == 0:
                  q.append(course)

          scheduled_courses = []
          while q:
              cur = q.popleft()
              scheduled_courses.append(cur)

              for nxt in g_outdegree[cur]:
                  g_indegree[nxt] -= 1
                  if g_indegree[nxt] == 0:
                      q.append(nxt)

          return scheduled_courses if numCourses == len(scheduled_courses) else []
          ```
    * 269: Alien Dictionary (H)
  * Other:
    * 685: Redundant Connection II
    * 323: Number of Connected Components in an Undirected Graph
### Bit Manipulation
  * Tips:
    * A^0 = A
      * **adding** "val" to the "set" if "val" is not in the "set"
    * A^A = 0
      * **removing** "val" from the "set" if "val" is already in the "set"
    * (ones ^ A[i]) & ~twos
      ```txt
      IF the set "ones" does not have A[i]
        Add A[i] to the set "ones" if and only if its not there in set "twos"
      ELSE
        Remove it from the set "ones"
      ```
    * x & -x
      * get the last set bit
      ```txt
      x = 1100
      -x = ^x + 1 = 0100
      x & -x = 0100
      ```
  * 268: Missing Number (E)
    * Description:
      * Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is **missing** from the array.
      * n number in n + 1 choice
    * Approach1: Sorting, Time:O(nlogn), Space:O(logn~n)
    * Approach2: Generate a sorted array-1, Time:O(n), Spaxe:O(n)
    * Approach3: Bit Manipulation (XOR), Time:O(n), Space:O(1)
      * example:
        ```txt
        idx: 0  1  2  3
        val: 0  1  3  4
        missing = 4 ^ (0^0) ^ (1^1) ^ (2^3) ^ (3^4)
                = 2 ^ (0^0) ^ (1^1) ^ (3^3) ^ (4^4)
                = 2 ^ 0
                = 2
        ```
      * Python
        ```python
        def missingNumber(self, nums: List[int]) -> int:
          missing = len(nums)
          for i, n in enumerate(nums):
              missing ^= i ^ n

          return missing
        ```
    * Approach4: Gauss formula, Time:O(n), Space:O(1)
      * Python
        ```python
        def missingNumber(self, nums: List[int]) -> int:
          expected_sum = (1 + len(nums)) * len(nums) // 2
          actual_sum = sum(nums)
          return expected_sum - actual_sum
        ```
  * 389: Find the Difference (E)
    * Description:
      * String t is generated by random shuffling string s and then add one more letter at a random position.
      * example:
        * s = "abcd"
        * t = "abcde"
    * Approach1: Sorting and Check, Time:O(m+n)log(m+n), Space:O(m+n)
      * Python
        ```python
        def findTheDifference(self, s: str, t: str) -> str:
          short, long = sorted(s), sorted(t)
          diff_idx = len(short)
          for i in range(len(short)):
              if short[i] != long[i]:
                  diff_idx = i
                  break

          return long[diff_idx]
        ```
    * Approach2: Counter, Time:O(m+n), Space:O(m)
    * Approach3: Bit manipulatoin, (XOR), Time:O(m+n), Space:O(1)
      * Python
        ```python
        def findTheDifference(self, s: str, t: str) -> str:
          res = 0

          for c in s:
              res ^= ord(c)

          for c in t:
              res ^= ord(c)

          return chr(res)
        ```
  * 136: Single Number (E)
    * Description:
      * Given a non-empty array of integers, **every element appears twice except for one.** Find that single one.
    * Approach1: Counter, Time:O(n), Space:O(n)
      * Python
        ```python
        def singleNumber(self, nums: List[int]) -> int:
          counter = collections.Counter(nums)
          res = None
          for n, cnt in counter.items():
              if cnt == 1:
                  res = n
                  break
          return res
        ```
    * Approach2: Bit manipulation, XOR, Time:O(n), Space:O(1)
      * Python
        ```python
        def singleNumber(self, nums: List[int]) -> int:
          res = 0
          for n in nums:
              res ^= n

          return res
        ```
  * 137: Single Number II (M)
    * Description:
      * Given a non-empty array of integers, **every element appears three times except for one**, which appears exactly once. Find that single one.
    * Approach1: Counter, Time:O(n), Space:O(n)
      * Python
        ```python
        def singleNumber(self, nums: List[int]) -> int:
          counter = collections.Counter(nums)
          res = None
          for c, cnt in counter.items():
              if cnt == 1:
                  res = c
                  break

          return res
        ```
    * Approach2: bit manipulation
      * Ref:
        * https://leetcode.com/problems/single-number-ii/discuss/43294/Challenge-me-thx
      * Concept:
        * **adding** "val" to the "set" if "val" is not in the "set" => A^0 = A
        * **removing** "val" from the "set" if "val" is already in the "set" => A^A = 0
        * **(ones ^ A[i]) & ~twos** means
          ```txt
          IF the set "ones" does not have A[i]
            Add A[i] to the set "ones" if and only if its not there in set "twos"
          ELSE
            Remove it from the set "ones"
          ```
        * **(twos^ A[i]) & ~ones** means
          ```txt
          IF the set "twos" does not have A[i]
            Add A[i] to the set "twos" if and only if its not there in set "ones"
          ELSE
            Remove it from the set "twos"
          ```
      * Python
        ```python
        def singleNumber(self, nums: List[int]) -> int:
          ones = twos = 0
          for n in nums:
              """
              1st appears
                  n in ones
                  n not in twos
              2nd appears
                  n not in ones
                  n in twos
              3rd appears
                  n not in ones
                  n in twos
              """
              ones = (ones ^ n) & (~twos)
              twos = (twos ^ n) & (~ones)
          return ones
        ```
  * 260: Single Number III (M)
    * Description:
      * Given an array of numbers nums, in which **exactly two elements appear only once** and all the other elements appear exactly twice. Find the two elements that appear only once.
    * Approach1: Counter, Time:O(n), Space:O(n)
      * Python
        ```python
        def singleNumber(self, nums: List[int]) -> List[int]:
          counter = collections.Counter(nums)
          ones = []
          for n, cnt in counter.items():
              if cnt == 1:
                  ones.append(n)
                  if len(ones) == 2:
                    break
          return ones
        ```
    * Approach2: Bit manipulation, Time:O(n), Space:O(1)
      * Ref:
        * https://leetcode.com/articles/single-number-iii/
      * Python
        ```python
        def singleNumber(self, nums: List[int]) -> List[int]:
          """
          get difference of x and y
          """
          diff = 0
          for n in nums:
              diff ^= n

          """
          get the last set bit, use this bit to distinguish x and y
          """
          diff &= -diff

          ones = [0, 0]
          for n in nums:
              if n & diff:
                  ones[0] ^= n
              else:
                  ones[1] ^= n
          return ones
        ```
### Dynamic Programming
  * Ref:
    * [From good to great. How to approach most of DP problems](https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.)
  * Note:
    * DP problems can be approached using the following sequence:
      * Recursive Relation
      * Recursive (top-down)
      * Recursive + memo (top-down)
      * Iterative + memo (bottom-up)
      * Iterative + N variables (bottom-up)
    * When handling DP problem, you should only focus current i when processing.
  * **Fibonacci sequence**:
    * 509: Fibonacci Number (E)
      * Recursive Relation
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
      * Given n will be a positive integer, each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
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
        * Python
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
        * Python
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
      * Approach4: Fibonacci Formula, Time:O(logn), Space:O(1)
        * Ref:
          * https://leetcode.com/problems/climbing-stairs/solution/
    * 746: Min Cost Climbing Stairs (E)
      * Recursive Relation:
        * f(0) == f(1) == 0
        * f(n) = min(f(n-1) + cost(n-1), f(n-2) + cost(n-2))
      * Approach1: Iterative + memo
        * Python
          ```python
          def minCostClimbingStairs(self, cost: List[int]) -> int:
            n = len(cost)

            if n <= 2:
                return 0

            memo = [0] * (len(cost)+1)

            for i in range(2, n+1):
                memo[i] = min(memo[i-1]+cost[i-1], memo[i-2]+cost[i-2])

            return memo[-1]
          ```
      * Approach2: Iterative + n variables
        * Python
          ```python
          def minCostClimbingStairs(self, cost: List[int]) -> int:
            n = len(cost)

            if n <= 2:
                return 0

            prev = cur = 0
            for i in range(2, n+1):
                prev, cur = cur, min(cur+cost[i-1], prev+cost[i-2])

            return cur
          ```
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
  * **Subsequence**:
    * 1143: Longest Common Subsequence, (LCS) (M)
      * Ref:
         * https://www.youtube.com/watch?v=NnD96abizww
         * https://leetcode.com/problems/longest-common-subsequence/discuss/348884/C%2B%2B-with-picture-O(nm)
      * Relation
        * definition:
          * memo[i][j] is LCS between s1[0:i+1] and s2:[0:j+1]
        * if s1[i] == s2[j]
          * memo[i][j] = memo[i-1][j-1] + 1
        * else (s1[i] != s2[j])
          * memo[i][j] = max(memo[i][j-1], memo[i-1][j])
      * DP Example:
        ```txt
        # x is dummy
          x a c e
        x 0 0 0 0
        a 0 1 1 1
        b 0 1 1 1
        c 0 1 2 2
        d 0 1 2 2
        e 0 1 2 3
        ```
      * Traverse Back Function:
        * Python
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
          ```
      * Approach1: DP: Time: O(mn), Space: O(mn)
        * Python
          * Implementation1:
            ```python
            def longestCommonSubsequence(self, text1: str, text2: str) -> int:

              s1, s2 = text1, text2
              l1, l2 = len(text1), len(text2)
              row, col = l1, l2

              if not l1 or not l2:
                  return 0

              memo = [[0 for _ in range(col)] for _ in range(row)]

              # init start
              if s1[0] == s2[0]:
                  memo[0][0] = 1

              # init first row
              for c in range(1, col):
                  if memo[0][c-1] or s2[c] == s1[0]:
                      memo[0][c] = 1

              # init first col
              for r in range(1, row):
                  if memo[r-1][0] or s1[r] == s2[0]:
                      memo[r][0] = 1

              # complete the memo
              for r in range(1, row):
                  for c in range(1, col):
                      if s1[r] == s2[c]:
                          memo[r][c] = memo[r-1][c-1] + 1
                      else:
                          memo[r][c] = max(memo[r][c-1], memo[r-1][c])

              return memo[row-1][col-1]
            ```
          * Implementation2 (waste some space but is more clear):
            ```python
            def longestCommonSubsequence(self, text1: str, text2: str) -> int:
              s1, s2 = text1, text2
              l1, l2 = len(text1), len(text2)
              row, col = l1+1, l2+1

              if not l1 or not l2:
                  return 0

              memo = [[0 for _ in range(col)] for _ in range(row)]

              # complete the memo
              for r in range(1, row):
                  for c in range(1, col):
                      if s1[r-1] == s2[c-1]:
                          memo[r][c] = memo[r-1][c-1] + 1
                      else:
                          memo[r][c] = max(memo[r][c-1], memo[r-1][c])

              return memo[row-1][col-1]
            ```
      * Approach2: DP + mem optimization1: O(mn), Space:O(n)
        * Ref:
          * https://leetcode.com/problems/longest-common-subsequence/discuss/348884/C%2B%2B-with-picture-O(nm)
        * Python
          ```python
          def longestCommonSubsequence(self, text1: str, text2: str) -> int:
            s1, s2 = text1, text2
            l1, l2 = len(text1), len(text2)
            row, col = l1 + 1, l2 + 1

            if not l1 or not l2:
                return 0

            # memory optimization, use two rows
            memo = [[0 for _ in range(col)] for _ in range(2)]

            for r in range(1, row):
                prev_r = (r - 1) % 2
                cur_r = r % 2
                for c in range(1, col):
                    if s1[r-1] == s2[c-1]:
                        memo[cur_r][c] = memo[prev_r][c-1] + 1
                    else:
                        memo[cur_r][c] = max(memo[cur_r][c-1], memo[prev_r][c])

            return memo[(row-1) % 2][-1]
          ```
      * Approach3: DP + mem optimization2: O(mn), Space:O(n)
        ```python
        def longestCommonSubsequence(self, text1: str, text2: str) -> int:
          s1, s2 = text1, text2
          l1, l2 = len(text1), len(text2)
          row, col = l1+1, l2+1

          if not l1 or not l2:
              return 0

          memo = [0 for _ in range(col)]

          # complete the memo
          for r in range(1, row):
              # use prev to keep memo[r-1][c-1]
              prev = memo[0]
              for c in range(1, col):
                  tmp = memo[c]
                  if s1[r-1] == s2[c-1]:
                      # memo[r][c] = memo[r-1][c-1]
                      memo[c] = prev + 1
                  else:
                      # memo[r][c] = max(memo[r][c-1], memo[r-1][c])
                      memo[c] = max(memo[c-1], memo[c])
                  prev = tmp

          return memo[-1]
        ```
    * 300: Longest Increasing Subsequence (LIS) (M)
      * see binary search
    * 673: Number of Longest Increasing Subsequence (LIS) (M)
      * Approach1: DP-1, Time:O(n^2), Space:O(n)
        * Python
          ```python
          def findNumberOfLIS(self, nums: List[int]) -> int:
            if not nums:
                return 0

            lis_memo = [0] * len(nums)
            cnt_memo = [0] * len(nums)
            max_lis = 1

            for j in range(len(nums)):
                lis = 1
                # 1. find the lip of end with position j
                for i in range(j):
                    if nums[j] > nums[i]:
                        lis = max(lis, 1+lis_memo[i])

                lis_memo[j] = lis
                max_lis = max(max_lis, lis)

                if lis == 1:
                    cnt_memo[j] = 1
                else:
                    # 2. update the lcs cnt with lis
                    for i in range(j):
                        if nums[j] > nums[i] and lis_memo[i] + 1 == lis:
                            cnt_memo[j] += cnt_memo[i]

            lis_cnt = 0
            for lis, cnt in zip(lis_memo, cnt_memo):
                if lis == max_lis:
                    lis_cnt += cnt

            return lis_cnt
          ```
      * Approach2: DP-2, Time:O(n^2), Space:O(n)
        * Python
          ```python
          def findNumberOfLIS(self, nums: List[int]) -> int:
            if not nums:
                return 0

            lis_memo = [1] * len(nums)
            cnt_memo = [1] * len(nums)
            max_lis = 1

            for j in range(len(nums)):
                lis = 1
                for i in range(j):
                    if nums[j] <= nums[i]:
                        continue
                    if lis_memo[i] + 1 > lis_memo[j]:
                        lis_memo[j] = 1 + lis_memo[i]
                        cnt_memo[j] = cnt_memo[i]
                    elif lis_memo[i] + 1 == lis_memo[j]:
                        cnt_memo[j] += cnt_memo[i]

                max_lis = max(max_lis, lis_memo[j])

            lis_cnt = 0
            for lis, cnt in zip(lis_memo, cnt_memo):
                if lis == max_lis:
                    lis_cnt += cnt

            return lis_cnt
          ```
      * Approach3: Segment Tree, Time:O(nlogn)
    * 674: Longest Continuous Increasing Subsequence (**LCIS**) (E)
      * Approach1: Iterative + memo, Time:O(n), Space:O(n)
        * Python
          ```python
          def findLengthOfLCIS(self, nums: List[int]) -> int:
            if not nums:
                return 0

            n = len(nums)
            memo = [1] * n
            max_lcis = 1

            for i in range(1, n):
                if nums[i] > nums[i-1]:
                    memo[i] = 1 + memo[i-1]
                    max_lcis = max(max_lcis, memo[i])

            return max_lcis
          ```
      * Approach2: Iterative + variables, Time:O(n), Space:O(1)
        * Python
          ```python
          def findLengthOfLCIS(self, nums: List[int]) -> int:
            if not nums:
                return 0

            n = len(nums)
            max_lcis = 1
            lcis = 1

            for i in range(1, n):
                if nums[i] > nums[i-1]:
                    lcis = 1 + lcis
                    max_lcis = max(max_lcis, lcis)
                else:
                    lcis = 1

            return max_lcis
          ```
  * **Best Time to Buy and Sell Stock**
    * Ref:
      * [General solution](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/discuss/108870/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems)
    * 121: Best Time to Buy and Sell Stock (E)
      * Description:
        * You may complete **at most one transaction**.
      * Approach1: Brute Force:O(n^2), Space:O(1)
      * Approach2: Time:O(n), Space:O(1)
        * For each round, keep the current minimum buy price and update best sell prcie.
        * Python:
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
      * Description
        * Multiple transcations allowed.
      * Approach1: Peak Valley , Time:O(n), Space:O(1)
        * Python:
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
        * Python Solution2
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            max_profit = 0
            for i in range(1, len(prices)):
                if prices[i] > price[i-1]:
                    max_profit += prices[i] - price[i-1]

            return max_profit
          ```
    * 714: Best Time to Buy and Sell **Stock with Transaction Fee** (M)
      * Description:
        * You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.
        * You may not buy more than 1 share of a stock at a time (ie. you must sell the stock share before you buy again.)
      * Solution:
        * Ref:
          * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/solution/
        * Cash(i):
          * The cash in hand, if you are **not holding the stock** at the end of day(i):
          * cash[i] = max(case1, case2, case3) = max(case1, case2)
            * case1:
              * cash[i] = cash[i-1]
            * case2:
              * cash[i] = hold[i-1] + prcie[i] - fee
            * case3:
              * cash[i] = chsh[i-1] - prices[i] - fee + prices[i] = cash[i-1] - fee
        * Hold(i):
          * The cash in hand, if you are **holding the stock** at the end of day(i):
          * hold[i] = max(case1, case2, case3) = max(case1, case2)
            * case1:
              * hold[i] = hold[i-1]
            * case2:
              * hold[i] = hold[i-1] + price[i] - fee - price[i]
            * case2:
              * hold[i] = hold[i-1] + prices[i] - fee - prices[i] = hold[i-1] - fee
      * Approach1: DP, Time:O(n), Space:O(n)
        * Python
          ```python
          def maxProfit(self, prices: List[int], fee: int) -> int:
            if not prices:
                return 0

            n = len(prices)
            cash = [0] * n
            hold = [0] * n
            hold[0] = -prices[0]

            for i in range(1, len(prices)):
                p = prices[i]
                cash[i] = max(cash[i-1], hold[i-1] + p - fee)
                hold[i] = max(hold[i-1], cash[i]-p)

            return cash[-1]
          ```
      * Approach2: DP, Time:O(n), Space:O(1)
        * Python
            ```python
            def maxProfit(self, prices: List[int], fee: int) -> int:
              if not prices:
                  return 0

              n = len(prices)
              cash = 0
              hold = -prices[0]

              for i in range(1, len(prices)):
                  p = prices[i]
                  prev_cash = cash
                  """
                  cash[i] = max(case1, case2)
                  case1 : cash[i-1]
                  case2:  hold[i-1] + prices[i] - fee
                  case3:  cash[i-1] - prices[i] - fee + prices[i]
                  """
                  cash = max(cash, hold+p-fee)
                  """
                  hold[i] = max(case1, case2, case3) = max(case1, case2)
                  case1: hold[i]
                  case2: cash[i-1] - prices[i]
                  case3: hold[i] + prices[i] - fee - prices[i]
                  """
                  hold = max(hold, prev_cash-p)

              return cash
            ```
    * 309: Best Time to Buy and Sell Stock with **Cooldown** (M)
      * Description:
        * You may complete as many transactions as you like
        * you must sell the stock before you buy again
        * After you sell your stock, you cannot buy stock on next day. (coolddown)
      * Solution:
        * Ref:
          * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/75928/Share-my-DP-solution-(By-State-Machine-Thinking)
        * Concept:
          * hold[i]: hold a stock, can sell
            * init: hold[0] = -prices[0]
            * hold[i] = max(case1, case2)
              * case1: hold[i-1]
              * case2: cash[i-1] - prices[i]
          * cooldown[i]: just sold stock, only can rest
            * init
            * cooldown[i] = hold[i-1] + prices[i]
          * cash[i]: do not hold any stock
            * init: 0
            * cash[i] = max(case1, case2)
              * case1: cash[i-1]
              * case2: cooldown[i-1]
      * Approach1: DP, Time:O(n), Space:O(n)
        * Python
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            if len(prices) <= 1:
                return 0

            n = len(prices)

            hold = [0] * n
            hold[0] = -prices[0]
            cash = [0] * n
            cooldown = [0] * n

            for i in range(1, n):
                p = prices[i]
                hold[i] = max(hold[i-1], cash[i-1]-p)
                cash[i] = max(cash[i-1], cooldown[i-1])
                cooldown[i] = hold[i-1] + p

            return max(cash[-1], cooldown[-1])
          ```
      * Approach2: DP, Time:O(n), Space:O(1)
        * Python
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            if len(prices) <= 1:
                return 0

            n = len(prices)

            hold = -prices[0]
            cash = 0
            cooldown = 0

            for i in range(1, n):
                p = prices[i]
                prev_cooldown = cooldown
                """
                cooldown[i] = hold[i-1] + p
                hold[i] = max(hold[i-1], cash[i-1]-p)
                cash[i] = max(cash[i-1], cooldown[i-1])
                """
                cooldown = hold + p
                hold = max(hold, cash-p)
                cash = max(cash, prev_cooldown)

            return max(cash, cooldown)
          ```
    * 123: Best Time to Buy and Sell Stock III (H)
      * Description:
        * You may complete **at most two transactions**.
      * Ref:
        * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/135704/Detail-explanation-of-DP-solution
      * Concept:
        * memo[k, i]: k transactions on i-th day
        * memo[k, i] = max(case1, case2)
          * case1: memo[k, i-1]: the profit is same as previous day
          * case2: max(prices[i] - prices[j] + memo[k-1][j]) for j=[0..i]
            * The best buying day from k-1 transactions
            * buy a stock on j-th day and sell on i-th day
            * When j == i, dp[k-1][i] looks like we just lose one chance of transaction.
      * Approach1: DP, Time:O(kn), Space:O(kn)
        * Python
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            if len(prices) <= 1:
                return 0

            max_trans = 2  # at most 2 transactions
            n = len(prices)
            memo = [[0 for _ in range(n)] for _ in range(max_trans+1)]

            """
            starting from k == 1, since profit of k == 0 is 0
            """
            for k in range(1, max_trans+1):
                """
                case1:
                    memo[k, i-1]: the profit is same as previous day
                case2:
                    prices[i] - prices[j] + memo[k-1][j]
                    buy a stock on j-th day and sell i-th day
                """
                best_buy = -prices[0]
                for i in range(1, n):
                    best_buy = max(best_buy, memo[k-1][i] - prices[i])
                    memo[k][i] = max(memo[k][i-1],
                                     prices[i] + best_buy)
            return memo[-1][-1]
          ```
      * Approach2-1: DP, Time:O(kn), Space:O(kn)
        * Scan col by col
        * Python
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            if len(prices) <= 1:
                return 0

            max_trans = 2  # at most 2 transactions
            n = len(prices)
            memo = [[0 for _ in range(n)] for _ in range(max_trans+1)]
            best_buy = [-prices[0]] * (max_trans+1)

            for i in range(1, n):
                for k in range(1, max_trans+1):
                    """
                    best_buy[k][i] = max(best_buy[k-1][i-1],
                                         memo[k-1][i] - prices[i])
                    """
                    best_buy[k] = max(best_buy[k], memo[k-1][i] - prices[i])
                    memo[k][i] = max(memo[k][i-1],
                                    prices[i] + best_buy[k])

            return memo[-1][-1]
          ```
      * Approach2-2: DP, Time:O(kn), Space:O(k)
        * Python
          ```python
          def maxProfit(self, prices: List[int]) -> int:
            if len(prices) <= 1:
                return 0

            max_trans = 2  # at most 2 transactions
            n = len(prices)
            memo = [0] * (max_trans+1)
            best_buy = [-prices[0]] * (max_trans+1)

            for i in range(1, n):
                for k in range(1, max_trans+1):
                    """
                    best_buy[k][i] = max(best_buy[k][i-1],
                                         memo[k-1][i] - prices[i])
                    """
                    best_buy[k] = max(best_buy[k],
                                      memo[k-1] - prices[i])

                    """
                    memo[k][i] = max(memo[k][i-1],
                                     prices[i] + best_buy[k]

                    """
                    memo[k] = max(memo[k],
                                  prices[i] + best_buy[k])

            return memo[-1]
          ```
    * 188: Best Time to Buy and Sell Stock IV (H)
      * Description:
        * You may complete **at most k transactions**.
      * Combination of 122 and 123.
        * if k >= n/2:
          * you can make maxium number of transactions (solution 122)
          * The reason why k becomes effectively unlimited is because we need at least 2 days to perform a sell - one day to buy and one day to sell.
        * else
          * solution 123
      * Approach1: DP, Time:O(n), Space:O(k)
        * Python
          ```python
          def maxProfit(self, k: int, prices: List[int]) -> int:

            def unlimited_transactions():
                profit = 0
                for i in range(1, len(prices)):
                    if prices[i] > prices[i-1]:
                        profit += (prices[i] - prices[i-1])

                return profit

            def limited_transactions(max_trans):
                n = len(prices)
                memo = [0] * (max_trans+1)
                best_buy = [-prices[0]] * (max_trans+1)

                for i in range(1, n):
                    for k in range(1, max_trans+1):
                        best_buy[k] = max(best_buy[k], memo[k-1] - prices[i])
                        memo[k] = max(memo[k], prices[i] + best_buy[k])

                return memo[-1]

            if len(prices) <= 1:
                return 0

            if k >= len(prices) / 2:
                return unlimited_transactions()
            else:
                return limited_transactions(max_trans=k)
          ```
  * **Triangle**:
    * 118: Pascal's Triangle (E)
      * Relation
        * if i == 0 or layer_num-1:
          * memo[k][i] = 1
        * else
          * memo[k][i] = memo[k-1][i] + memo[k-1][i-1]
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
      * Relation
        * if i == 0 or layer_num-1:
          * memo[k][i] = 1
        * else
          * memo[k][i] = memo[k-1][i] + memo[k-1][i-1]
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
                    """
                    pascal[r][i] = pascal[r-1][i-1] + pascal[r-1][i]
                    """
                    pascal[i] += pascal[i-1]

            return pascal
          ```
    * 120: Triangle (M)
      * Relation
        * Starting from top layer
          * if i == 0 or i == layer_num - 1:
            * memo[k][i] = 1
          * else
            * memo[k][i] = triangle[k][i] + memo[k-1][i-1]
        * Starting from bottom layer
          * Ref:
            * https://leetcode.com/problems/triangle/discuss/38730/DP-Solution-for-Triangle
          * memo[k][i] = triangle[k+1][i] + memo[k+1][i+1]
      * Approahc1: Starting from Top layer, Time:O(mn), Space:O(mn)
        * Python
          ```python
          def minimumTotal(self, triangle: List[List[int]]) -> int:
            layer = len(triangle)

            if layer == 1:
                return triangle[0][0]

            memo = []

            # init first row
            memo.append(triangle[0])

            # starting from 2nd row
            for l in range(1, layer):
                cnt = l + 1
                new_layer = [0] * cnt
                new_layer[0] =  triangle[l][0] + memo[-1][0]
                new_layer[-1] = triangle[l][-1] + memo[-1][-1]
                for i in range(1, cnt-1):
                    new_layer[i] = triangle[l][i] + min(memo[-1][i], memo[-1][i-1])

                memo.append(new_layer)

            return min(*memo[-1])
          ```
      * Approach2: Starting from Top layer, Time:O(mn), Space:O(n)
        * Python
          ```python
          def minimumTotal(self, triangle: List[List[int]]) -> int:
            layer = len(triangle)

            if layer == 1:
                return triangle[0][0]

            memo = [0] * layer

            # init first row
            memo[0] = triangle[0][0]

            # starting from 2nd row
            for l in range(1, layer):

                # for each row, process from end to beginning
                cnt = l + 1

                # cnt - 1
                memo[cnt-1] = triangle[l][cnt-1] + memo[cnt-2]
                # fron cnt-2 to 1
                for i in range(cnt-2, 0, -1):
                    memo[i] = triangle[l][i] + min(memo[i], memo[i-1])
                # 0
                memo[0] += triangle[l][0]

            return min(*memo)
          ```
      * Approach3: Starting from bottom layer, Time:O(mn), Space:O(n)
        * Python
          ```python
          def minimumTotal(self, triangle: List[List[int]]) -> int:
            layer = len(triangle)

            # copy the bottom layer
            memo = triangle[-1][:]

            # from layer-2 to 0
            for l in range(layer-2, -1, -1):
                cnt = l + 1
                for i in range(cnt):
                    memo[i] = triangle[l][i] + min(memo[i], memo[i+1])

            return memo[0]
          ```
  * **One Dimensional**:
    * 624: Maximum Distance in Arrays (E)
      * Description:
        * Given m arrays, and each array is sorted in ascending order. Now you can pick up two integers from two different arrays (each array picks one) and calculate the distance. We define the distance between two integers a and b to be their absolute difference |a-b|. Your task is to find the maximum distance.
      * Approach1: Brute Force, Time:O(n^2), Space:O(1)
        * Cnsider only the distances between the first(minimum element) element of an array and the last(maximum element) element of the other arrays and find out the maximum distance from among all such distances.
        * Python
          ```python
          def maxDistance(self, arrays: List[List[int]]) -> int:
            a = arrays
            n = len(a)
            max_dis = float('-inf')

            for i in range(0, n-1):
                for j in range(i+1, n):
                    max_dis = max(max_dis, abs(a[i][0] - a[j][-1]))
                    max_dis = max(max_dis, abs(a[j][0] - a[i][-1]))
            return max_dis
          ```
      * Approach2: Single Scan, Time:O(n^2), Space:O(1)
        * Keep the pre min and prev max
        * Python
          ```python
          def maxDistance(self, arrays: List[List[int]]) -> int:
            a = arrays
            min_val, max_val = a[0][0], a[0][-1]
            max_dis = 0

            for i in range(1, len(a)):

                cur_min, cur_max = a[i][0], a[i][-1]
                max_dis = max(max_dis,
                              abs(max_val-cur_min),
                              abs(cur_max-min_val))

                # update min and max for next list
                min_val = min(min_val, cur_min)
                max_val = max(max_val, cur_max)

            return max_dis
          ```
    * 055: Jump Game (M)
      * Description:
        * Given an array of non-negative integers, you are initially positioned at the first index of the array.
        * Each element in the array represents your maximum jump length at that position.
        * Determine if you are able to reach the last index.
      * Ref:
        * https://leetcode.com/problems/jump-game/solution/
      * Approach1: DP, Recursive + memo (top-down), Time: O(n^2), Space: O(n)
        * Python
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
        * Python
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
        * Python
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
    * 062: Unique Paths (M)
      * The robot **can only move either down or right** at any point in time.
      * Approach1: Combination:
        * total m-1 down steps and n-1 right steps
        * (m+n)!/m!n!
        * Python
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
        * Python
          ```python
          def uniquePaths(self, m: int, n: int) -> int:
            memo = [ [1 for _ in range(n)] for _ in range(m)]

            for r in range(1, m):
                for c in range(1, n):
                    memo[r][c] = memo[r][c-1]  + memo[r-1][c]

            return memo[-1][-1]
          ```
      * Approach3: DP, Time:O(mn), Space:O(n)
        * In fact, keep one row is enough
        * Python
          ```python
          def uniquePaths(self, m: int, n: int) -> int:
            if not m or not n:
                return 0

            memo = [1 for _ in range(n)]
            for _ in range(1, m):
                for c in range(1, n):
                    """
                    memo[r][c] = memo[r-1][c] + memo[r][c-1]
                    memo[c]    = memo[c]      + memo[c-1]
                    """
                    memo[c] += memo[c-1]

            return memo[-1]
          ```
    * 063: Unique Paths II (M)
      * Now consider if some obstacles are added to the grids. How many unique paths would there be?
      * Approach1: Time:O(mn), Space:O(mn)
        * Python
          ```python
          def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
            if obstacleGrid[0][0]:
                return 0

            row = len(obstacleGrid)
            col = len(obstacleGrid[0])

            memo = [[0 for _ in range(col)] for _ in range(row)]

            # init first row
            memo[0][0] = 1
            for c in range(1, col):
                if obstacleGrid[0][c]:
                    break
                memo[0][c] = 1

            # init first col
            for r in range(1, row):
                if obstacleGrid[r][0]:
                    break
                memo[r][0] = 1

            for r in range(1, row):
                for c in range(1, col):
                    if not obstacleGrid[r][c]:
                        memo[r][c] = memo[r-1][c] + memo[r][c-1]

            return memo[-1][-1]
          ```
      * Approach2: Time:O(mn), Space:O(n)
        * Python
          ```python
          def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
            if obstacleGrid[0][0]:
                return 0

            row = len(obstacleGrid)
            col = len(obstacleGrid[0])

            memo = [0 for _ in range(col)]

            # init first row
            memo[0] = 1
            for c in range(1, col):
                if obstacleGrid[0][c]:
                    break
                memo[c] = 1

            for r in range(1, row):

                if obstacleGrid[r][0]:
                    memo[0] = 0

                for c in range(1, col):
                    if obstacleGrid[r][c]:
                        memo[c] = 0
                    else:
                        """
                        memo[r][c] = memo[r-1][c] + memo[r][c-1]
                        memo[c]    = memo[c]      + memo[c-1]
                        """
                        memo[c] += memo[c-1]

            return memo[-1]
          ```
    * 139: Word Break (M)
      * Example1:
        * intput:
          *  s = "leetcode", wordDict = ["leet", "code"]
        * output:
          * true
      * Example2:
        * input:
          * s = "applepenapple", wordDict = ["apple", "pen"]
        * output:
          * true
      * Note:
        * The same word in the dictionary may be reused multiple times in the segmentation.
        * You may assume the dictionary does not contain duplicate words.
      * Complexity Analysis
        * Assume the complexity for substring comparison is O(1)
      * Approach1: Recursive + memo, Time:O(n^2), Space:O(n)
        * algo:
          * check every possible prefix of that string in the dictionary of words, if it is found in the dictionary, then the recursive function is called for the remaining portion of that string.
        * complexity:
          * Time:O(n^2)
            * total n positions in the memo, and each postion needs (n + 1 to end) time substring comparison
          * Space: O(n)
        * Python
          ```python
          def wordBreak(self, s: str, wordDict: List[str]) -> bool:

            def backtrack(start):
                if start == len(s):
                    return True

                if memo[start] is not None:
                    return memo[start]

                memo[start] = False
                for end in range(start, len(s)):
                    if (end - start + 1) > max_word_len:
                        break

                    prefix = s[start:end+1]
                    if prefix in word_d and backtrack(start=end+1):
                      memo[start] = True
                      break

                return memo[start]


            max_word_len = 0
            word_d = dict()
            for w in wordDict:
                word_d[w] = True
                max_word_len = max(max_word_len, len(w))

            memo = [None] * len(s)

            backtrack(start=0)

            return memo[0]
          ```
      * Approach2: Iterative + memo, Time:O(n^2), Space:O(n)
        * Same idea as approach1
        * Python
          ```python
          def wordBreak(self, s: str, wordDict: List[str]) -> bool:

            max_word_len = 0
            word_d = dict()
            for w in wordDict:
                word_d[w] = True
                max_word_len = max(max_word_len, len(w))

            visited = [False] * len(s)

            stack = []
            stack.append(0)
            res = False

            while stack:
                start = stack.pop()

                if start == len(s):
                    res = True
                    break

                if visited[start]:
                    continue

                visited[start] = True

                for end in range(start, len(s)):
                    # check word len first
                    if (end - start + 1) > max_word_len:
                        break
                    prefix = s[start:end+1]
                    if prefix in word_d:
                        stack.append(end+1)

            return res
          ```
      * Approach3: DP, Time:O(n^2), Space:O(n)
        * Python
          * Space:n + 1 implementation
            ```python
            def wordBreak(self, s: str, wordDict: List[str]) -> bool:
              if not s:
                  return True

              max_word_len = 0
              word_d = dict()
              for w in wordDict:
                  word_d[w] = True
                  max_word_len = max(max_word_len, len(w))

              """
              memo[i] means s[0:i] can be composed by words in word_d.
              """
              memo = [False] * (len(s) + 1)
              memo[0] = True

              for end in range(1, len(s)+1):
                  memo[end] = False
                  for start in range(0, end):
                      """
                      handleing the substring from 0 to end
                      each string is separated two parts prefix and suffix
                      since memo[0] is True, we can do whole suffix matching
                      """
                      # check prefix in memo first
                      #
                      if memo[start]:
                          # check suffix string start + 1 to end
                          # end - (start + 1) + 1
                          if (end - start) > max_word_len:
                            continue

                          # transform index s[start+1: end+1] to s[start: end]
                          suffix = s[start: end]
                          if suffix in word_d:
                              memo[end] = True
                              break

              return memo[-1]
            ```
          * Space n implementation:
            ```python
            def wordBreak(self, s: str, wordDict: List[str]) -> bool:

              def is_in_word_d(start, end):
                  if (end - start + 1) > max_word_len:
                      return False

                  sub_str = s[start:end+1]
                  return True if sub_str in word_d else False

              if not s:
                  return True

              max_word_len = 0
              word_d = dict()
              for w in wordDict:
                  word_d[w] = True
                  max_word_len = max(max_word_len, len(w))

              """
              memo[i] means s[0:i+1] can be composed by words in word_d.
              """
              memo = [False] * len(s)
              if s[0] in word_d:
                  memo[0] = True


              for end in range(1, len(s)):
                  """
                  handleing the substring from 0 to end
                  1. check whole substring
                  2. separate substring to prefix and suffix
                  """
                  # 1.
                  memo[end] = is_in_word_d(0, end)
                  if memo[end]:
                      continue

                  # 2.
                  for start in range(0, end):
                      # check prefix in memo first
                      if memo[start]:
                          # check suffix, start + 1 to end
                          memo[end] = is_in_word_d(start+1, end)
                          if memo[end]:
                              break

              return memo[-1]
            ```
    * 279: Perfect Squares (M)
    * 375: Guess Number Higher or Lower II (M)
    * 312: Burst Balloons (H)
  * **Two Dimensional**:
    * 072: Edit Distance (H)
      * Ref:
        * https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and-intuition
        * https://leetcode.com/problems/edit-distance/discuss/25846/C%2B%2B-O(n)-space-DP
      * Recursive relation
        * if word1[i] == word[j]  # no edit
          * memo[i][j] = memo[i-1][j-1]
        * elif word1[i] =! word[j]:
          * memo[i][j] = minimum of
            * case1: **R**eplace a character in word1
              * 1 + memo[i-1][j-1]
            * case2: **D**elete a character from word1 (insert a character into word2)
              * 1 + memo[i-1][j]
            * case3: **I**nsert a character into word1 (delete a character from word2)
              * 1 + memo[i][j-1]
      * Note:
        * insert a dummy character will not affect the minimum edit distance.
          * example:
            * (#horse, #ros) == (horse, ros)
      * DP Example:
        ```txt
        # x is dummy
          x   a         b          d
        x 0   1         2          3
        a 1   0(a)      1(Ib, ab)  2(Id, abd)
        b 2   1(Db, a)  0(ab)      1(Id, abd)
        c 3   2(Dc, a)  1(Dc, ab)  1(Ec, abd or abc)
        c 4   3(Dc, a)  2(Ib, abb) 2(Dc, abd) or (Ec, abd)
        ```
      * Approach1: Recursive (top-down)
        * Python
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
      * Approach2: DP, Time:O(mn), Space:O(mn)
        * Python
          ```python
          def minDistance(self, word1: str, word2: str) -> int:
            s1, s2 = word1, word2
            l1, l2 = len(word1), len(word2)
            row, col = l1 + 1, l2 + 1

            if not l1 and not l2:
                return 0

            if not l2:
                return l1

            if not l1:
                return l2

            memo = [[0 for _ in range(col)] for _ in range(row)]

            # init first row
            # assume there is a dummy character before word1 and word2
            for c in range(col):
                memo[0][c] = c

            # init first col
            for r in range(row):
                memo[r][0] = r

            for r in range(1, row):
                for c in range(1, col):
                    if word1[r-1] == word2[c-1]:
                        memo[r][c] = memo[r-1][c-1]
                    else:
                        """
                        1. edit:    memo[r-1][c-1]
                        2. delete:  memo[r-1][c]
                        3. insert:  memo[r][c-1]
                        """
                        memo[r][c] = 1 + min(memo[r-1][c-1],
                                            memo[r-1][c],
                                            memo[r][c-1])

            return memo[row-1][col-1]
          ```
      * Approach3: DP + mem optimization1, Time:O(mn), Space:O(n)
        * Python
          ```python
          def minDistance(self, word1: str, word2: str) -> int:
            s1, s2 = word1, word2
            l1, l2 = len(word1), len(word2)
            row, col = l1 + 1, l2 + 1

            if not l1 and not l2:
                return 0

            if not l2:
                return l1

            if not l1:
                return l2

            memo = [[0 for _ in range(col)] for _ in range(2)]

            # init first row
            # assume there is a dummy character before word1 and word2
            for c in range(col):
                memo[0][c] = c

            for r in range(1, row):
                prev_r = (r -1) % 2
                cur_r = r % 2
                memo[cur_r][0] = r
                for c in range(1, col):
                    if word1[r-1] == word2[c-1]:
                        memo[cur_r][c] = memo[prev_r][c-1]
                    else:
                        """
                        1. edit:    memo[r-1][c-1]
                        2. delete:  memo[r-1][c]
                        3. insert:  memo[r][c-1]
                        """
                        memo[cur_r][c] = 1 + min(memo[prev_r][c-1],
                                                memo[prev_r][c],
                                                memo[cur_r][c-1])

            return memo[(row-1)%2][col-1]
          ```
      * Approach4: DP + mem optimization2, Time:O(mn), Space:O(n)
        * Python
          ```python
          def minDistance(self, word1: str, word2: str) -> int:
            s1, s2 = word1, word2
            l1, l2 = len(word1), len(word2)
            row, col = l1 + 1, l2 + 1

            if not l1 and not l2:
                return 0

            if not l2:
                return l1

            if not l1:
                return l2

            memo = [0 for _ in range(col)]

            # init first row
            # assume there is a dummy character before word1 and word2
            for c in range(col):
                memo[c] = c

            for r in range(1, row):
                prev = memo[0]
                memo[0] = r
                for c in range(1, col):

                    tmp = memo[c]

                    if word1[r-1] == word2[c-1]:
                        # memo[c] = memo[r-1][c-1]
                        memo[c] = prev
                    else:
                        """
                        1. edit:    memo[r-1][c-1] = prev
                        2. delete:  memo[r-1][c]   = memo[c]
                        3. insert:  memo[r][c-1]   = memo[c-1]
                        """
                        memo[c] = 1 + min(prev,
                                          memo[c],
                                          memo[c-1])

                    prev = tmp

            return memo[-1]
          ```
    * 256: Paint House (E) (L)
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
        * Python
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
        * Python:
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
      * Approach1: DP, Time:O(mn), Space:O(mn)
        * Python
          ```python
          def minPathSum(self, grid: List[List[int]]) -> int:
            row, col = len(grid), len(grid[0])
            memo = [[0 for _ in range(col)] for _ in range(row)]
            memo[0][0] = grid[0][0]

            # init first row
            for c in range(1, col):
                memo[0][c] = memo[0][c-1] + grid[0][c]

            # init first col
            for r in range(1, row):
                memo[r][0] = memo[r-1][0] + grid[r][0]

            # complete the memo
            for r in range(1, row):
                for c in range(1, col):
                    memo[r][c] = grid[r][c] + min(memo[r-1][c], memo[r][c-1])

            return memo[-1][-1]
          ```
      * Approach2: DP with mem optimization, Time:O(mn), Space:O(n)
        * Python
          ```python
          def minPathSum(self, grid: List[List[int]]) -> int:
            row, col = len(grid), len(grid[0])
            memo = [0 for _ in range(col)]
            memo[0] = grid[0][0]

            # init first row
            for c in range(1, col):
                memo[c] = grid[0][c] + memo[c-1]

            # complete the memo
            for r in range(1, row):
                # memo[r][0] = grid[r][0] + memo[r-1][0]
                memo[0] += grid[r][0]

                for c in range(1, col):
                    # memo[r][c] = grid[r][c] + min(memo[r-1][c], memo[r][c-1])
                    memo[c] = grid[r][c] + min(memo[c], memo[c-1])

            return memo[-1]
          ```
    * 097: Interleaving String (H)
    * 174: Dungeon Game (H)
    * 221: Maximal Square (M)
    * 085: Maximal Rectangle (H)
    * 363: Max Sum of Rectangle No Larger Than KTreeSe (H)
    * 741: Cherry Pickup (H)
  * **Deduction**:
    * 714: Best Time to Buy and Sell Stock with Transaction Fee (M)
    * 276: Paint Fence (E) (L)
      * Description
        * There is a fence with **n posts**, each post can be painted with one of the **k colors**.
        * You have to paint all the posts such **that no more than two adjacent fence posts have the same color**.
        * Return the total number of ways you can paint the fence.
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
        * Python
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
        * Python
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
                """
                same[i] = diff[i-1]
                diff[i] = (diff[i-1] + same[i-1]) * (k-1)
                """
                same, diff = diff, (same + diff) * (k-1)

            return diff + same
    * House Robber
      * 198: House Robber (E)
        * Description:
          * You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed
          * The only constraint stopping you from robbing each of them is that **adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night**.
        * Recursive relation:
          * version1
            * For n <= 2
              * f(0) = 0
              * f(1) = nums[0]
              * f(2) = max(nums[0], nums[1])
            * For n > 2
              * f(n) = max(case1, case2)
              * case1: rob house n: monery(n) + f(n-2)
              * case2: do not rob house n: f(n-1)
          * version2:
            * For n < 1:
              * f(0) = 0
              * f(1): rob, not_rob = nums[0], 0
            * For n > 1:
              * f(n)
                * rob[n] = nums[n] + not_rob[n-1]
                * not_rob[n = max(rob[n-1], not_bod[n-1])
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
        * Iterative + N variables (bottom-up) v1, Time:O(n), Space:(1)
          * Python
          ```python
          def rob(self, nums: List[int]) -> int:
            n = len(nums)

            if n == 0:
                return 0

            if n == 1:
                return nums[0]

            # n == 2 (2th house)
            prev, cur = nums[0], max(nums[0], nums[1])

            # starting from 3th house
            for i in range(2, n):
                prev, cur = cur, max(nums[i] + prev, cur)

            return cur
            ```
        * Iterative + N variables (bottom-up) v2, Time:O(n), Space:(1)
          * Python
            ```python
            def rob(self, nums: List[int]) -> int:
              n = len(nums)

              if n == 0:
                  return 0

              rob = nums[0]
              not_rob = 0
              if n == 1:
                  return nums[0]

              # starting from 2th house
              for i in range(1, n):
                  """
                  rob[i] = nums[i] + not_rob[i-1]
                  not_rob[i] = max(not_rob[i-1], rob[i-1])
                  """
                  rob, not_rob = nums[i] + not_rob, max(rob, not_rob)

              return max(rob, not_rob)
            ```
      * 213: House Robber II (M)
        * Description
          * All houses at this place are arranged **in a circle**.
        * How to break the circle ?
          * (1) i not robbed
          * (2) i robbed
          * (3) pick the bigger one.
        * Python
          ```python
          def rob(self, nums: List[int]) -> int:
            def rob_non_cycle(start, end):
                """
                198: House Robber
                """
                rob, not_rob = nums[start], 0
                for i in range(start+1, end+1):
                    rob, not_rob = nums[i] + not_rob, max(rob, not_rob)

                return max(rob, not_rob)


            n = len(nums)

            if n == 0:
                return 0
            if n == 1:
                return nums[0]
            if n == 2:
                return max(nums[0], nums[1])
            if n == 3:
                return max(nums[0], nums[1], nums[2])

            """
            n >= 4
            case1: rob the 0th house and skip 1th hourse and (n-1)th house
            case2: do not rob the 0th
            """
            rob_first = nums[0] + rob_non_cycle(2, n-2)
            do_not_rob_first = rob_non_cycle(1, n-1)

            return max(rob_first, do_not_rob_first)
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
  * **Knapsack Problem:**
    * 322: Coin Change (M)
      * Description:
        * Write a function to compute the **fewest number of coins that you need to make up that amount**.
      * Ref:
        * https://leetcode.com/problems/coin-change/discuss/77368/*Java*-Both-iterative-and-recursive-solutions-with-explanations
      * Example:
        * 1
          * Input: coins = [1, 2, 5], amount = 11
          * Output: 3
        * 2
          * Input: coins = [2], amount = 3
          * Output: -1
      * Approach1: backtracking:
        * Python
          ```python
          def coinChange(self, coins: List[int], amount: int) -> int:
            def backtrack(target, start, cur_num):
                nonlocal min_num

                if target < 0:
                    return

                if target == 0:
                    min_num = min(cur_num, min_num)
                    return

                # combination, prevent duplicated
                for i in range(start, len(coins)):
                    backtrack(target-coins[i], i, cur_num+1)

            min_num = float('inf')
            backtrack(amount, 0, 0)

            if min_num == float('inf'):
                min_num = -1

            return min_num
          ```
      * Recursive Relation:
        * f(n) = 1 + min(f[i-coin1], f[i-coin2], ... f[i-coink])
        * else not found
      * Approach2: Top down + memo, Time:O(S*n), Space:O(S)
        * Python
          ```python
          def coinChange(self, coins: List[int], amount: int) -> int:
            def coin_change(target):
                if memo[target] != None:
                    return memo[target]

                memo[target] = not_found
                min_cnt = float('inf')
                for coin in coins:
                    if coin > target:
                      break
                    cnt = coin_change(target-coin)
                    if cnt != not_found:
                        min_cnt = min(min_cnt, 1 + cnt)

                if min_cnt != float('inf'):
                    memo[target] = min_cnt

                return memo[target]

            not_found = -1
            coins.sort()
            memo = [None] * (amount + 1)
            memo[0] = 0
            coin_change(target=amount)
            return memo[-1]
          ```
      * Approach3: Bottom up DP, Time:O(S*n), Space:O(S)
        * Python
          ```python
          def coinChange(self, coins: List[int], amount: int) -> int:
            not_found = -1
            coins.sort()
            memo = [not_found] * (amount+1)
            memo[0] = 0

            for target in range(1, amount+1):
                min_cnt = float('inf')
                for coin in coins:
                    if coin > target:
                        break

                    res = memo[target-coin]
                    if res != not_found:
                      min_cnt = min(min_cnt, 1+res)

                if min_cnt != float('inf'):
                    memo[target] = min_cnt

            return memo[-1]
          ```
    * 983: Minimum Cost For Tickets (M)
      * Description:
        * The passes allow that many days of consecutive travel.  For example, if we get a 7-day pass on day 2, then we can travel for 7 days: day 2, 3, 4, 5, 6, 7, and 8.
      * Ref:
        * https://leetcode.com/problems/minimum-cost-for-tickets/discuss/226659/Two-DP-solutions-with-pictures
      * Recursive Relation:
        * f(n) = min(cost1+f(n-1), cost7+f(n-7), cost30++f(n-30))
      * Approach1: Track Calendar days, Time:O(N), Space:O(N)
        * N is the number of calendar days
        * Python
          ```python
          def mincostTickets(self, days: List[int], costs: List[int]) -> int:
            travel_days = set(days)
            max_day = days[-1]
            memo = [0] * (max_day + 1)

            for day in range(1, max_day+1):
                if day not in travel_days:
                    memo[day] = memo[day-1]
                else:
                    memo[day] = min(costs[0] + memo[day-1],
                                    costs[1] + memo[max(0, day-7)],
                                    costs[2] + memo[max(0, day-30)])
            return memo[-1]
          ```
      * Approach2: Track Traverl days, Time:O(n), Space:O(38)
        * n is the number of traverl days
          * Python
            ```python
            import collections

            TOP = 0
            DAY = 0
            COST = 1

            def mincostTickets(self, days: List[int], costs: List[int]) -> int:
                last_7 = collections.deque()
                last_30 = collections.deque()
                cost = 0

                for day in days:
                    # pop expire
                    while last_7 and last_7[TOP][DAY] + 6 < day:
                        last_7.popleft()

                    while last_30 and last_30[TOP][DAY] + 29 < day:
                        last_30.popleft()

                    last_7.append((day, cost))
                    last_30.append((day, cost))

                    cost = min(cost + costs[0],
                               last_7[TOP][COST] + costs[1],
                               last_30[TOP][COST] + costs[2])

                return cost

              ```
    * 518: Coin Change 2 (M)
      * Description
        * Write a function to compute the number of combinations that make up that amount.
      * Ref:
        * https://leetcode.com/problems/coin-change-2/discuss/99212/Knapsack-problem-Java-solution-with-thinking-process-O(nm)-Time-and-O(m)-Space
    * 039: Combination Sum
      * see backtracking
    * 416: Partition Equal Subset Sum (M)
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
    * Edit Distance:
      * all
    * **Sliding Window**
    * KMP
    * **Palindrome**
      * all execept 125: valid Palindrome (E)
    * **Parentheses**
      * 020: Valid Parentheses (E)
      * 022: Generate Parentheses (M)
    * **Subsequence**
      * 392: Is Subsequence (E)
      * 1143: Longest Common Subsequence (M)
      * 300: Longest Increasing Subsequence (**LIS**) (M)
    * Reorder:
      * 186: Reverse **Words** in a String II (M)
      * 557: Reverse **Words** in a String III (E)
      * 345:	Reverse **Vowels** of a String (E)
    * Encode(Compression) and Decode
      * 038: Count and Say (E)
      * 443: String Compression (E)
    * Isomorphism
    * Anagram
  * Array
    * Check Duplicate
      * 220: Contains Duplicate III (M)
      * 287: Find the Duplicate Number (M)
    * Remove Duplicate
    * **Containers**
    * Jump Game
    * **Best Time to Buy and Sell Stock**
    * Shortest Word Distance
    * **Interval**
      * all
    * **Subarray**
      * all
    * Number
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
  * Stack and Queue
    * 155: Min Stack (E)
    * 716: Max Stack (E)
    * 394: Decode String (M)
    * 255: Verify Preorder Sequence in Binary Search Tree (M)
    * 341: Flatten Nested List Iterator (M)
    * 536: Construct Binary Tree from String (M)
    * Calculator
  * Tree:
    * PreOrder:
      * Same Tree
          * 100: **Same** Tree (E)
          * 572: Subtree of Another Tree (E)
      * Path sum
      * 298: Binary Tree **Longest Consecutive** Sequence (M)
    * InOrder:
      * 094: Binary Tree **Inorder** Traversal (M)
      * 105: Construct Binary Tree from **Preorder** and **Inorder** Traversal (M)
      * 106: Construct Binary Tree from **Inorder** and **Postorder** Traversal	(M)
    * PostOrder:
      * 250: Count **Univalue Subtrees** (M)
      * 366: Find Leaves of Binary Tree (M)
      * **Lowest Common Ancestor** of a **Binary Tree**
      * Path
        * 124: Binary Tree **Maximum Path Sum** (H)
        * 687: Longest **Univalue Path** (E)
        * 549: Binary Tree Longest Consecutive Sequence II (M)
    * Level Order:
      * 107: Binary Tree Level Order Traversal II
      * 199: Binary Tree **Right Side** View
      * 314: Binary Tree **Vertical Order** Traversal	(M)
      * 116 & 117: **Populating Next Right Pointers** in Each Node (M)
    * Binary Tree Slots
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
    * 034: Find **First and Last Position of Element** in Sorted Array (M)
    * 033 & 081: Search in Rotated Sorted Array (M)
    * 153 & 154: Find Minimum in **Rotated Sorted Array**
    * 004: Median of Two Sorted Arrays (H)
    * 222: Count **Complete** Tree Nodes (M)
    * 300: Longest Increasing Subsequence (**LIS**)
  * Cache
    * 146: LRU Cache (M)
    * 460: LFU Cache (H)
  * Trie (Prefix Tree)
    * 211: Add and Search Word - Data structure design (M)
  * Heap
    * Merge k Sorted Lists
      * 023: Merge k Sorted Lists (H)
      * 378: **Kth Smallest Element** in a Sorted Matrix (M)
      * 373: **Find K Pairs** with Smallest Sums (M)
    * Schedule
      * 253: Meeting Rooms II (M)
      * 1094: Car Pooling (M)
      * 1109: Corporate Flight Bookings (M)
    * Kth
      * 703: **Kth Largest** Element in a Stream
      * 215: **Kth Largest** Element in an Array
        * quick select
      * 973: K Closest Points to Origin
    * 295: Find **Median** from Data Stream (H)
  * BackTracking
    * 022: Generate Parentheses (M)
    * 077: Combinations (M)
    * 078: Subsets (M)
  * BFS & DFS
    * **Islands**
      * 200: Number of Islands (M)
      * 695: Max Area of Island (M)
    **Nested**
       * 364: **Nested List** Weight Sum II (M)
    * **Nested**
      * 339: Nested List Weight Sum (E)
      * 565: Array Nesting (M)
    * 286: Walls and Gates (M)
      * Start From Gates
    * Surrounded Regions (M)
    * 127: Word Ladder (M)
  * Union Find
  * Graph
    * Eulerian trail
      * 332: Reconstruct Itinerary (M)
    * Topological Sort
      * 207: Course Schedule (M)
  * Dynamic Programming:
    * Fibonacci sequence
      * 070: Climbing Stairs
      * 091: **Decode Ways**
    * Triangle:
      * 120: Triangle (M)
    * **One Dimensional**
      * 062: Unique Paths (M)
      * 139: Word Break (M)
    * **Two Dimensional**
      * 1143: Longest Common Subsequence (M)
        * Memory Optimizatino version Space:O(n)
      * 072: Edit Distance (H)
        * Notice the init condition
        * Memory Optimizatino version Space:O(n)
      * 256: Paint House (E)
      * 064: Minimum Path Sum (M)
    * Knapsack problem
      * 322: Coin Change (M)
      * 983: Minimum Cost For Tickets (M)
      * 518: Coin Change 2 (M)
    * **Deduction**
      * 276: Paint Fence (E)
      * 198: House Robber (E)
      * 213: House Robber II (M)
      * 337: House Robber III (M)
  * Bit Manipulation
