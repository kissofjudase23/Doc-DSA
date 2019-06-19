Table of Contents
- [Reference](#Reference)
- [Data Structures](#Data-Structures)
  - [General Tips](#General-Tips)
  - [Linked List](#Linked-List)
  - [Stack](#Stack)
  - [Queue](#Queue)
  - [Hash Table](#Hash-Table)
  - [Cache](#Cache)
    - [LRU Cache](#LRU-Cache)
  - [Tree](#Tree)
- [Algorithms](#Algorithms)
  - [Search](#Search)
    - [Time Complexity](#Time-Complexity)
    - [Implementation](#Implementation)
  - [Sort](#Sort)
    - [Time Complexity](#Time-Complexity-1)
    - [Implementation](#Implementation-1)
    - [Quick Sort](#Quick-Sort)
    - [Merge Sort](#Merge-Sort)
  - [Insertion Sort](#Insertion-Sort)
  - [Selection Sort](#Selection-Sort)
  - [Bubble](#Bubble)

## Reference
* Complexity
  * [bigocheatsheet](http://bigocheatsheet.com/)
* [Leetcode](https://leetcode.com/)
  * [Classification](https://cspiration.com/leetcodeClassification#103)
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
* [Codeforces](https://codeforces.com/)
  * If you can achieve **1800 scores within 1 hour** that would be highly recommended for Google applicants.
* [Interviewbit](https://www.interviewbit.com/courses/programming/)
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
* [Cracking the coding interview](http://www.crackingthecodinginterview.com/)
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


## [Data Structures](https://github.com/kissofjudase23/Library-python-common-modules/tree/master/common/ds)

### General Tips
* The **"Stack"** can help for:
  * Palindrome issues
  * DFS
* The **"Queue"** can help for:
  * BFS

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
  * Doubly Linked List + Hash Table

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | LRU Cache  | O(1) | O(1) | O(1)   |

### Tree
* Time complexity
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

### Insertion Sort
  * Best Case: ascending sequence, [1, 2, 3]
  * Worst Case: descending sequence, [3, 2, 1]
### Selection Sort
  * Unstable: [5, 5*, 3]
### Bubble
  * Best Case:
    * ascending sequence, [1, 2, 3]
    * (n-1) comparison in round1, and no swap happened.
  * Worst Case:
    * descending sequence, [3, 2, 1]