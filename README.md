Table of Contents
- [Reference](#Reference)
  - [Complexity](#Complexity)
  - [Leetcode](#Leetcode)
    - [Classification](#Classification)
  - [Codeforces](#Codeforces)
  - [Interviewbit](#Interviewbit)
  - [Cracking the coding interview](#Cracking-the-coding-interview)
  - [FAQ](#FAQ)
- [Data Structures](#Data-Structures)
  - [Array](#Array)
  - [String](#String)
  - [Matrix](#Matrix)
  - [Linked List](#Linked-List)
  - [Stack](#Stack)
  - [Queue](#Queue)
  - [Hash Table](#Hash-Table)
  - [Cache](#Cache)
    - [LRU Cache](#LRU-Cache)
  - [Tree](#Tree)
    - [Time complexity](#Time-complexity)
    - [Binary Tree](#Binary-Tree)
    - [Binary Heaps (Min-Heaps and Max-Heaps)](#Binary-Heaps-Min-Heaps-and-Max-Heaps)
    - [Tries (Prefix Trees)](#Tries-Prefix-Trees)
    - [AVL Tree](#AVL-Tree)
    - [Red-Black Tree](#Red-Black-Tree)
    - [B Tree](#B-Tree)
  - [Graph](#Graph)
    - [Represent](#Represent)
    - [Search](#Search)
- [Algorithms](#Algorithms)
  - [Search](#Search-1)
    - [Time Complexity](#Time-Complexity)
    - [Implementation](#Implementation)
  - [Sort](#Sort)
    - [Time Complexity](#Time-Complexity-1)
    - [Implementation](#Implementation-1)
    - [Quick Sort](#Quick-Sort)
    - [Merge Sort](#Merge-Sort)
    - [Insertion Sort](#Insertion-Sort)
    - [Selection Sort](#Selection-Sort)
    - [Bubble Sort](#Bubble-Sort)
  - [Backtracking](#Backtracking)
  - [Dynamic Programming](#Dynamic-Programming)
  - [Greddy](#Greddy)
  - [Topoligical Sort](#Topoligical-Sort)
- [Others](#Others)
  - [Bit Mamipulation](#Bit-Mamipulation)


## Reference
### Complexity
  * [bigocheatsheet](http://bigocheatsheet.com/)
### [Leetcode](https://leetcode.com/)
#### [Classification](https://cspiration.com/leetcodeClassification#103)
  * Array
  * String
  * Math
  * Tree
  * Backtracking
  * Dynamic Programming
  * LinkedList
  * Binary Search
  * Matrix
  * DFS & BFS
  * Stack & Priority Queue
  * Bit Mamipulation
  * Topological Sort
  * Random
  * Graph
  * Union Find
  * Trie
  * Design
### [Codeforces](https://codeforces.com/)
  * If you can achieve **1800 scores within 1 hour** that would be highly recommended for Google applicants.
### [Interviewbit](https://www.interviewbit.com/courses/programming/)
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
### [Cracking the coding interview](http://www.crackingthecodinginterview.com/)
  * Data Structures
    * Arrays and Strings
    * Linked List
    * Stacks and Queues
    * Tree and Graphs
  * Concepts and Algorithms
    * Bit Manipulation
    * Math and Logic Puzzles
    * Object-Oriented Design
    * System Design
    * Sorting and Searching
### FAQ
  * [What is tail recursion?](https://stackoverflow.com/questions/33923/what-is-tail-recursion)


## [Data Structures](https://github.com/kissofjudase23/Library-python-common-modules/tree/master/common/ds)

### Array

### String

### Matrix

### Linked List
* Tips
  * The **"Runner"** Techinique (or second pointer)
    * The runner techinique means that you iterate through the linked list with two pointers simultaneously, with one head of the other.
    * Loop issues, find the middle, etc.

* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/linkedlist.py)
* Time complexity
  * Average and Worst

    |                              | Search | Push_front | Push_back | Remove Node |
    |------------------------------|--------|------------|-----------|-------------|
    | Singly Linked List with tail | O(n)   | O(1)       | O(1)      | O(n)        |
    | Doubly Linked List with tail | O(n)   | O(1)       | O(1)      | O(1)        |

* LeetCode
  * Tips:
    * The **"Runner"** Techinique
    * The **"dummy node"** Techinique
      * **dummy.next** will be the new head node
    * Use **reverse** instead of **stack** for space complexity reduction.
      * However, reverse will change the data of the input, use it carefully.

 * Reverse
    * 206: Reverse Linked List (E)
    * 92: Reverse Linked List II (M) *
      * From **position m to n**. Do it in **one-pass**.
       * Using **"dummy node"**.
 * Circle
    * Use the **"Runner"** Techinique
    * 141: Linked List **Cycle** (E)
    * 142: Linked List Cycle II (M) *
      * Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
  * Remove
    * 237: Delete Node in a Linked List (E)
    * 19: Remove Nth Node From End of List (M)
      * Use the **"Runner"** Techinique
    * 203:	Remove Linked List Elements (E)
    * 83: Remove Duplicates from Sorted List (M)
  * Reorder *
    * Using the **"dymmy node"** Techinique
    * Swap Nodes in Pair (M) *
      * Using the **"dymmy node"** Techinique
      * Use 3 pointers, prev ,current and next
    * 328: Odd Even Linked List (M)
      * Create **two linked lists** and **merge** them
    * 143: Reorder List (M) *
      * Space O(1): *
        * Use the **"Runner"** Techinique
        * Reverse the last half of the linked list
        * Combine the first half and last half
      * Space O(n):
        * use a stack to store last half of the linked list
    * 148: sort list *
      * For constance space complexity
        * Use the **merge sort** algorithm (iterative version)
          * Having to implement split and merge funcitons
          * Having to handle linking issue between two sorted lists
    * *61: Rotate Lis
      * The rotate length k may be greater than the linked list length

  * 2: Add Two Numbers (M)
    * When using one loop
      * don't forget the last carry. *
  * 160	Intersection of Two Linked Lists (E)
    * Use difference of length
  * 21: Merge Two Sorted Lists (E)
    * The concept is like merge step in the merge sort.
  * 234: Palindrome Linked List(M)
    * Space O(1) *:
      * Reverse first half of the linked list, but it is not a pratical solution since we should not modity the constant function of the input.
    * Space O(n):
      * use a stack






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

* LeetCode
  * [146, LRU Cache](https://leetcode.com/problems/lru-cache/) (M)

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
    ```python
      def in_order_traversal(node):

        if not node:
          return

        in_order_traversal(node.left)
        visit(node)
        in_order_traversal(node.right)
    ```
  * Pre-Order
    ```python
      def pre_order_traversal(node):

        if not node:
          return

        visit(node)
        pre_order_traversal(node.left)
        pre_order_traversal(node.right)
    ```
  * Post-Order
    ```python
      def post_order_traversal(node):

        if not node:
          return

        post_order_traversal(node.left)
        post_order_traversal(node.right)
        visit(node)
    ```
* LeetCode
  * 144: Binary Tree Preorder Traversal (M)
    * Use **one stack** for iterative method
  * 94: Binary Tree Inorder Traversal (M)
    * Use **one stack** for iterative method
  * 145: Binary Tree Postorder Traversal (H)
    * Use **two stacks** for iterative method
  * 102: Binary Tree Level Order Traversal (M)
    * Use **the length of the queue** for each round

#### Binary Heaps (Min-Heaps and Max-Heaps)
* Ref:
  * https://towardsdatascience.com/data-structure-heap-23d4c78a6962
* **complete binary tree**
* Min-Heaps
  * Ascencding order
* Max-Heaps
  * Descending Order
* Insert
  * **bubble up** Operation which takes **O(log(n))**
  * Insert at the **rightmost spot so as to maintain the complete binary tree**.
  * **Fix the tree by swapping the new element with parents**, until finding the appropriate spot.
* Extract minimum (maximum) elements
  * **bubble down** which takes **O(log(n))**
  * Remove the mimimum element and swap it with the last element in the heap.
  * Bubble down this element, swapping it with one of its children until the min-heap property is restored.

#### Tries (Prefix Trees)
* Ref:
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

* LeetCode
  * [208: Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
    * medium
    * serach word and search prefix
  * [211: Add and Search Word - Data structure design](https://leetcode.com/problems/add-and-search-word-data-structure-design/)
    * medium
    * search word (support wildcard)
      * For iterative methods
        * Use **stack** for traversal, and **use continue instead of return false when not found**.
  * 212: Word Search II
    * hard


#### AVL Tree
* Balanced Tree

#### Red-Black Tree
* Balanced Tree

#### B Tree
* Balanced Tree

### Graph
#### Represent
  * Adjacency List
  * Adjacency Matrices
#### Search
  * Depth-First Search (DFS)
  * Breadth-First Search (BFS)


## Algorithms
### Search
#### Time Complexity

  |                  | Average     | Worst      | Space Worst |
  |------------------|-------------|------------|-------------|
  | Binary Search (I)| O(long(n))  | O(long(n)) | O(1)        |
  | Binary Search (R)| O(long(n))  | O(long(n)) | O(long(n))  |


#### [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/algo/search.py)

### Sort
#### Time Complexity

  |                | Best       | Average    | Worst      | Space Complexity | Stable/Unstable |
  |----------------|------------|------------|------------|------------------|-----------------|
  | Quick Sort     | O(nlog(n)  | O(nlog(n)  | O(n^2)     | O(log(n))        | Unstable        |
  | Merge Sort     | O(nlog(n)  | O(nlog(n)  | O(nlog(n)  | O(n)             | Stable          |
  | Heap Sort      | O(nlog(n)  | O(nlog(n)  | O(nlog(n)  | O(1)             | Unstable        |
  | Insertion Sort | O(n)       | O(n^2)     | O(n^2)     | O(1)             | Stable          |
  | Selection Sort | O(n^2)     | O(n^2)     | O(n^2)     | O(1)             | Unstable        |
  | Bubble Sort    | O(n)       | O(n^2)     | O(n^2)     | O(1)             | Stable          |
  | Radix Sort     | O(d*(n+r)) | O(d*(n+r)) | O(d*(n+r)) | O(r*n)           | Stable          |

#### [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/algo/search.py)
#### Quick Sort
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

    * T(n) = T(n-1) + cn

  * Unstable Case:
    * [quicksort algorithm stability](https://stackoverflow.com/questions/13498213/quicksort-algorithm-stability)
    * [4, 2, 1 ,4*, 3]
      * round1:
        * partition:
          * pivot: 3
          * [2, 1, 3 ,4*, 4]
  * Recursive
    * https://www.geeksforgeeks.org/python-program-for-quicksort/
    * https://www.youtube.com/watch?v=CB_NCoxzQnk

  * Iterative
    * https://www.techiedelight.com/iterative-implementation-of-quicksort/

#### Merge Sort
  * Ref
    * https://www.geeksforgeeks.org/merge-sort/
    * https://www.youtube.com/watch?v=6pV2IF0fgKY

  * Recursive:
    * ![flow](https://www.darkwiki.in/wp-content/uploads/2017/12/merge-sort-working-in-hindi-with-example-1.png)
  * Iterative:
    * ![flow](https://images.slideplayer.com/25/7830874/slides/slide_19.jpg)

#### Insertion Sort
  * Best Case: ascending sequence, [1, 2, 3]
  * Worst Case: descending sequence, [3, 2, 1]
#### Selection Sort
  * Unstable: [5, 5*, 3]
#### Bubble Sort
  * Best Case:
    * ascending sequence, [1, 2, 3]
    * (n-1) comparison in round1, and no swap happened.
  * Worst Case:
    * descending sequence, [3, 2, 1]

### Backtracking


### Dynamic Programming

### Greddy


### Topoligical Sort

## Others
### Bit Mamipulation
