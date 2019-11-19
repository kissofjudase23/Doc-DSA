Table of Contents
- [FAQ](#faq)
- [Walking through a Problem](#walking-through-a-problem)
- [Optimize & Solve Technique](#optimize--solve-technique)
- [Codeforces](#codeforces)
- [Complexity](#complexity)
- [Data Structures](#data-structures)
  - [Linked List](#linked-list)
  - [Stack](#stack)
  - [Queue](#queue)
  - [Hash Table](#hash-table)
  - [Cache](#cache)
    - [LRU Cache](#lru-cache)
    - [LFU Cache](#lfu-cache)
  - [Tree](#tree)
    - [Time complexity](#time-complexity)
    - [Binary Tree](#binary-tree)
    - [Binary Heaps (Min-Heaps and Max-Heaps)](#binary-heaps-min-heaps-and-max-heaps)
    - [Tries (Prefix Trees)](#tries-prefix-trees)
    - [AVL Tree](#avl-tree)
    - [Red-Black Tree](#red-black-tree)
    - [B Tree](#b-tree)
  - [Graph](#graph)
- [Algorithms](#algorithms)
  - [Search](#search)
  - [Sort](#sort)
  - [Dynamic Programming](#dynamic-programming)
    - [Recursion and Dynamic Programming](#recursion-and-dynamic-programming)
    - [Dynamic Programming & Memorization](#dynamic-programming--memorization)


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



## Complexity
  * [bigocheatsheet](http://bigocheatsheet.com/)
  * [What is a plain English explanation of “Big O” notation?](https://stackoverflow.com/questions/487258/what-is-a-plain-english-explanation-of-big-o-notation)


## [Data Structures](https://github.com/kissofjudase23/Library-python-common-modules/tree/master/common/ds)

### Linked List
* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/linkedlist.py)
* Time complexity
  * Average and Worst

    |                              | Search | Push_front | Push_back | Remove Node |
    |------------------------------|--------|------------|-----------|-------------|
    | Singly Linked List with tail | O(n)   | O(1)       | O(1)      | O(n)        |
    | Doubly Linked List with tail | O(n)   | O(1)       | O(1)      | O(1)        |

### Stack
* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/stack.py)
* Time complexity
  * Average and Worst

      |       | Push | Pop  | Search|
      |-------|------|------|-------|
      | Stack | O(1) | O(1) | O(n)  |

### Queue
* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/queue.py)
* Time complexity
  * Average and Worst

    |       | Add  | Remove  | Search|
    |-------|------|---------|-------|
    | Queue | O(1) | O(1)    | O(n)  |


### Hash Table
* Time complexity
  * Average

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | Hash Table | O(1) | O(1) | O(1)   |

  * Worst

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | Hash Table | O(n) | O(n) | O(n)   |



### Cache
#### LRU Cache
* ![Flow](https://blog.techbridge.cc/img/kdchang/cs101/algorithm/lru-algorithm.png)
* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/cache.py)
* Time complexity
  * Use **Doubly linked List** as underlying data structure for O(1) node remove operation
  * Use **hash table** for O(1) lookup
  * Doubly Linked List + Hash Table

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | LRU Cache  | O(1) | O(1) | O(1)   |

#### LFU Cache
### Tree
#### Time complexity
  * Average

    |                    | Access     | Search     | Insertion  | Deletion   |
    |--------------------|------------|------------|------------|------------|
    | Binary Search Tree | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | AVL Tree           | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | B Tree             | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | Red-Black Tree     | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |

  * Worst

    |                    | Access     | Search     | Insertion  | Deletion   |
    |--------------------|------------|------------|------------|------------|
    | Binary Search Tree | O(n)       | O(n)       | O(n)       | O(n)       |
    | AVL Tree           | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | B Tree             | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | Red-Black Tree     | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |

#### Binary Tree
* Types
  * Binary search Trees:
    * **All left descendents <= n < all right descendents**
  * Complete Binary Trees
    * A binary tree in which every level of the tree is **fully filled**, except the rightmost element on the last level.
  * Full Binary Trees
    * A binary tree in which every node has **either zero of two children**.
  * Perfect Binary Trees
    * A binary tree is one that **full and complete**.
    * All leaf nodes will be at the same level, and this level has the maximum number of nodes.

* Traversal
  * In-Order
    * recursive
        ```python
        def in_order_traversal(node):
          if not node:
            return

          in_order_traversal(node.left)
          visit(node)
          in_order_traversal(node.right)
        ```
    * iterative

        ```python
        def in_order_traversal(node):

          current = node
          stack = list()

          while current or stack:

            if current:
              stack.append(current)
              current = current.left

            else:
              node = stack.pop()
              visit(node)
              current = current.right
        ```
  * Pre-Order
    * recursive
      ```python
        def pre_order_traversal(node):

          if not node:
            return

          visit(node)
          pre_order_traversal(node.left)
          pre_order_traversal(node.right)
      ```
    * iterative
      ```python
        def pre_order_traversal(node):

          if not node:
            return

          current = node
          stack = list()

          while current or stack:
            if not current:
              current = stack.pop()

            visit(current)
            if current.right
              stack.append(current.right)
            current = current.left
      ```
  * Post-Order
    * recursive
      ```python
        def post_order_traversal(node):

          if not node:
            return

          post_order_traversal(node.left)
          post_order_traversal(node.right)
          visit(node)
      ```
    * iterative
      ```python
        def post_order_traversal(node):

          if not node:
            return

          s1 , s2 = list(), list()
          stack.append(node)

          while s1:
            node = s1.pop()
            if node.left:
              s1.append(node.left)

            if node.right:
              s1.append(node.right)

          while s2:
            node = s2.pop()
            visit(node)
      ```


#### Binary Heaps (Min-Heaps and Max-Heaps)
* Ref:
  * https://towardsdatascience.com/data-structure-heap-23d4c78a6962
  * [MIT OpenCourseWare](https://www.youtube.com/watch?v=B7hVxCmfPtM)
* **Complete Binary Tree**
* Min-Heaps
  * Ascencding order
* Max-Heaps
  * Descending Order
* Array Implementation:
  * Index
    * Left Child
      * 2i + 1
    * Right Child
      * 2i + 2
    * Parent
      * (i - 1) // 2

* **Heapify**
  * Time: **O(n)**
  * Create a heap from an array, build from index n/2 to 1 (skip the last level)
    ```python
      def build_min_heap(array)
        # from n/2 downto 1
        for i in range(len(array)//2, 0, -1)
            min_heapify(array, i)
    ```
* **Insert**
  * Time: **O(log(n))**
    * **bubble up** Operation
  * Insert at the **rightmost spot so as to maintain the complete binary tree**.
  * **Fix the tree by swapping the new element with parents**, until finding the appropriate spot.
* **Extract minimum (maximum) elements**
  * Time: **O(log(n))**
    * **bubble down** Operation
  * Remove the mimimum element and swap it with the last element in the heap.
  * Bubble down this element, swapping it with one of its children until the min-heap property is )restored.

#### Tries (Prefix Trees)
* Reference:
  * https://leetcode.com/articles/implement-trie-prefix-tree/
  * https://www.youtube.com/watch?v=AXjmTQ8LEoI
  * https://www.youtube.com/watch?v=zIjfhVPRZCg

* A trie is a variant of an n-ary tree in which characters are stored at each node. Each path down the tree may represent a word.
* Used to store collections of words.
  * If 2 words have a common prefix then they will have the same ancesctors in this trie.
  * abcl
  * abgd
  * ![image](./image/ds/tries.png)

* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/trie.py)
  ```python
    class TrieNode(object):
      def __init__(self):
          self.children = dict()
          self.end_of_word = False
  ```
* Use cases:
  * **Prefix** lookups.
  * **Whole word** lookups.
  * Autocomplete
  * Spell checker

* FAQ
  * Compare with Hash Table
    * While a hash table can quickly loop up whether a string is a valid word, **it cannot tell us if a string is a prefix of any words**.
    * A trie can check if a string is a valid prefix in O(k), where k is the length of the string. Although we often refer to hash table loopups as being O(1) time, this isn't entirely true. **A hash table must read though all the characters in the input**, which takes **O(k)** time in the case of a word lookup.


#### AVL Tree
* Balanced Tree

#### Red-Black Tree
* Balanced Tree

#### B Tree
* Balanced Tree

### Graph
* Representation
  * Complexity
    * Incidence (vertex to edge)

    |                  | Storage        | Add Vertex     | Add Edge       | Remove Vertex  | Remove Edge |
    |------------------|----------------|----------------|----------------|----------------|-------------|
    | Adjacency List   | O(\|V\|+\|E\|) | O(1)           | O(1)           | O(\|V\|+\|E\|) | O(\|E\|)    |
    | Incidence List   | O(\|V\|+\|E\|) | O(1)           | O(1)           | O(\|E\|)       | O(\|E\|)    |
    | Adjacency Matrix | O(\|V\|^2)     | O(\|V\|^2)     | O(1)           | O(\|V\|^2)     | O(1)        |
    | Incidence Matrix | O(\|V\|*\|E\|) | O(\|V\|*\|E\|) | O(\|V\|*\|E\|) | O(\|V\|*\|E\|) | O(\|E\|)    |

* Algorithm:
  * Traverse
    * BFS
      * Need Queue Support
    * DFS
      * Need Stack Support
  * Euler trail
    * A finite graph that **visits every edge** exactly once.
  * Euler circuit
    * An **Eulerian trail** that **starts and ends on the same vertex**.
  * Hamilton path
    * A finite graph that **visits every vertex** exactly once.
  * Hamilton cycle
    * An Hamilton path that **starts and ends on the same vertex**.
  * Topological Sort

    |                  | Average Time   | Worst Time     | Space          |
    |------------------|----------------|----------------|----------------|
    | Topological sort | O(\|V\|+\|E\|) | O(\|V\|+\|E\|) | O(\|V\|+\|E\|) |

  * Minimum Spanning Tree (Greedy)
    * A minimum spanning tree (MST) or minimum weight spanning tree is a subset of the edges of a connected, edge-weighted undirected graph that connects all the vertices together, without any cycles and with the minimum possible total edge weight.

    |                     | Average Time     | Worst Time | Space          |
    |---------------------|------------------|------------|----------------|
    | Kruskal's algorithm | O(\|E\|log\|V\|) |            |                |
    | Prim's algorithm    | O(\|E\|log\|V\|) | O(\|V\|^2) | O(\|V\|+\|E\|) |
    | Sollin's algorithm  |                  |            |                |

  * Shortest Path:

    |                               | Average Time     | Worst Time    | Space Complexity |
    |-------------------------------|------------------|---------------|------------------|
    | Dijkstra's algorithm (Greedy) | O(\|E\|log\|V\|) | O(\|V\|^2)    | O(\|V\|+\|E\|)   |
    | A* search algorithm           | O(\|E\|)         | O(b^d)        | O(b^d)           |
    | Bellman–Ford algorithm (DP)   | O(\|E\|\|V\|)    | O(\|E\|\|V\|) | O(\|E\|)         |
    | Floyd-Warshall algorithm (DP) | O(\|V\|^3)       | O(\|V\|^3)    | O(\|V\|^2)       |


## Algorithms
### Search
  * Time Complexity

    |                  | Average     | Worst      | Space Worst |
    |------------------|-------------|------------|-------------|
    | Binary Search    | O(long(n))  | O(long(n)) | O(1)        |

  * [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/algo/search.py)

### Sort
  * Time Complexity

    |                | Best       | Average    | Worst      | Space Complexity | Stable/Unstable |
    |----------------|------------|------------|------------|------------------|-----------------|
    | Quick Sort     | O(nlog(n)  | O(nlog(n)  | O(n^2)     | O(log(n))        | Unstable        |
    | Merge Sort     | O(nlog(n)  | O(nlog(n)  | O(nlog(n)  | O(n)             | Stable          |
    | Heap Sort      | O(nlog(n)  | O(nlog(n)  | O(nlog(n)  | O(1)             | Unstable        |
    | Insertion Sort | O(n)       | O(n^2)     | O(n^2)     | O(1)             | Stable          |
    | Selection Sort | O(n^2)     | O(n^2)     | O(n^2)     | O(1)             | Unstable        |
    | Bubble Sort    | O(n)       | O(n^2)     | O(n^2)     | O(1)             | Stable          |
    | Counting Sort  | O(n+k)     | O(n+k)     | O(n+k)     | O(n+k)           | Stable          |
    | Bucket Sort    | O(n)       | O(n+k)     | O(n^2)     | O(1)             | Stable          |
    | Radix Sort     | O(d*(n+k)) | O(d*(n+k)) | O(d*(n+k)) | O(n+k)           | Stable          |

  * [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/algo/sort.py)

  * Quick Sort
    * FAQ
      * [Why does QuickSort use O(log(n)) extra space?](https://stackoverflow.com/questions/12573330/why-does-quicksort-use-ologn-extra-space)
        * To get rid of the recursive call you would have to use **a stack** in your code, and it would still occupy **log(n)** space.
      * [Quick sort implement by queue?](https://stackoverflow.com/questions/39666714/quick-sort-implement-by-queue)
        * Queue method requires O(n) space for sorting an array of size n.
    * Worst Case:
      * [1, 2, 3, 4, 5]
        * round1:
          * pivot: 5
          * left partition [1, 2, 3, 4]
          * right partition []
        * round2:
          * pivot: 4
          * left partition [1, 2, 3]
          * right partition []
        * ...
      * Solution:
        * Use median of three to pick pivot
    * Unstable:
      * [quicksort algorithm stability](https://stackoverflow.com/questions/13498213/quicksort-algorithm-stability)
      * [4, 2, 1 ,4*, 3]
        * round1:
          * partition:
            * pivot: 3
            * [2, 1, 3 ,4*, 4]
    * Related topic:
      * [Quick Search ith element in an array](https://leetcode.com/problems/kth-largest-element-in-an-array/solution/)

* Merge Sort
  * Ref
    * https://www.geeksforgeeks.org/merge-sort/
    * https://www.youtube.com/watch?v=6pV2IF0fgKY
  * Recursive:
    * Top-Down
  * Iterative:
    * Bottom Up

* Insertion Sort
  * Best Case: ascending sequence, [1, 2, 3]
  * Worst Case: descending sequence, [3, 2, 1]

* Selection Sort
  * Unstable: [5, 5*, 3]

* Bubble Sort
  * Best Case:
    * ascending sequence, [1, 2, 3]
    * (n-1) comparison in round1, and no swap happened.
  * Worst Case:
    * descending sequence, [3, 2, 1]

* Heap Sort
  * Build the Max Heap first
    * It takes O(n)
  * Iteratve n-1 round, for i in range(n-1, 0, -1)
    * Swap the current max(0th index) to the ith index
    * heapify from 0 to i-1

* Radix Sort:
  * The subsort for radix sort should be stable sort (like counting sort or bucket sort)

### Dynamic Programming
#### Recursion and Dynamic Programming
   * Recursive
     * By definition, are built of solutions to subproblems.
     * **Bottom-Up** Approach
       * It often the most intuitive.
     * **Top-Down** Approach
       * The top-down solution can be more complex since it's less concrete.
     * **Half-and-Half** Approach
       * Like merge sort and binary search
     * Recursive vs. Iteration Solutions
       * Recursive algorithms can be very **space inefficient**. Each recursive call adds a new layer to the stack.
       * For this reason, it's often better to implement a recurisve algorithm iteratively.

#### Dynamic Programming & Memorization
  * Dynamic Programming is mostly just a matter of taking a **recursive algorithm** and **finding the overlapping subproblems**. You then cache those results for future recursive calls.
  * This particular problem and most of others can be approached using the following sequence:
    * Find recursive relation
    * Recursive (top-down)
    * Recursive + memo (top-down)
    * Iterative + memo (bottom-up)
    * Iterative + N variables (bottom-up)
  * Example:
    * resursive: O(n^2), O(n)
      ```python
      def fib(i):
        if i == 0 or i == 1:
            return i

        return fib(i-1) + fib(i-2)
      ```
    * Iterative: O(n), O(1)
      ```python
      def fib(i):
         if i == 0 or i == 1:
            return i

          a, b = 0, 1
          for _ in range(2, i+1):
            a, b = b, a + b

          return b
      ```
    * **Top-Down** Dynamic Programming (or Memoization): O(n), O(n)
      ```python
      def fib(i):

        memo = dict()

        def _fib(i):
          if i == 0 or i == 1:
            return i

          if i not in memo:
            memo[i] = _fib(i-1), _fib(i-2)

          return memo[i]

        return _fib(i)

      ```
    * **Bottom-Up** Dynamic Programming: O(n), O(n)
      ```python
      def fib(i):

        if i == 0 or i == 1:
            return i

        memo = dict()
        memo[0] = 0
        memo[1] = 1

        for f in range(2, i+1):
          memo[f] = memo[f-1] + memo[f-2]

        return memo[i]
      ```



