from .base import TreeNode, NTreeNode
from typing import List

class MiddleTopic(TreeNode):
	def listOfDepth(self, tree: TreeNode) -> List[ListNode]:
		'''
		desc: 特定深度节点链表
		给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表
		（比如，若一棵树的深度为 D，则会创建出 D 个链表）。
		返回一个包含所有深度的链表的数组。
		'''
        def dfs(tree):
            if not tree:
                return []
            stack = [tree]
            new_stack = []
            n = len(stack)
            output = []
            tmp = []
            while stack:
                while n>0:
                    node = stack.pop(0)
                    tmp.append(node.val)
                    if node.left:
                        new_stack.append(node.left)
                    if node.right:
                        new_stack.append(node.right)
                    n -= 1
                cur = head = ListNode(None)
                for i in tmp:
                    head.next = ListNode(i)
                    head = head.next
                output.append(cur.next)
                tmp = []
                stack = new_stack
                new_stack = []
                n = len(stack)
            return output
        return dfs(tree)

    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
    	'''
		desc: 最大二叉树
		给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
			二叉树的根是数组中的最大元素。
			左子树是通过数组中最大值左边部分构造出的最大二叉树。
			右子树是通过数组中最大值右边部分构造出的最大二叉树。
			通过给定的数组构建最大二叉树，并且输出这个树的根节点。
    	'''
    	def dfs(nums):
            if not nums:
                return None
            sorted_nums = [(nums[i], i) for i in range(len(nums))]
            sorted_nums = sorted(sorted_nums, key=lambda x:x[0], reverse=True)
            merge = TreeNode(sorted_nums[0][0])
            merge.left = dfs(nums[:sorted_nums[0][1]])
            merge.right = dfs(nums[sorted_nums[0][1]+1:])
            return merge
        return dfs(nums)

    def searchTreeLeafNode(self, root: TreeNode) -> list:
        '''
        寻找二叉树的叶子节点
        找到叶子节点，删除，在剩下的树中继续查找叶子节点
        只要找到所有节点的高度，即可生成这样的结果数组。
        '''
        if not root:
            return None
        
        res = []
        def dfs(node):
            if not node:
                return 0
            depth = max(dfs(node.left), dfs(node.right))
            if len(res) < depth + 1:
                res.append([])
            res[depth].append(node.val)
 
            return depth + 1
        dfs(root)
        return res

    def insertIntoMaxTree(self, root: TreeNode, val: int) -> TreeNode:
        '''
        desc: 最大二叉树 II
        '''
        if not root:
            return TreeNode(val)
        if root.val<val:
            node = TreeNode(val)
            node.left = root
            return node
        else:
            root.right = self.insertIntoMaxTree(root.right, val)
            return root
        return None

    def longestConsecutive(root: TreeNode) -> int:
        '''
        desc: 二叉树最长连续序列
        '''
        def dfs(root, n):
            if not root:
                return
            if root.left and abs(root.left.val - root.val)==1:
                left = n + 1
                self.m = max(self.m, left)
                dfs(root.left, left)
            else:
                dfs(root.left, 1)
            if root.right and abs(root.right.val - root.val)==1:
                right = n + 1
                self.m = max(self.m, right)
                dfs(root.right, left)
            else:
                dfs(root.right, 1)

        if not root:
            return 0 
        self.m = 1
        dfs(root, 1)
        return self.m

    def countPairs(self, root: TreeNode, distance: int) -> int:
        '''
        desc: 好叶子节点对的数量
        给你二叉树的根节点 root 和一个整数 distance 。
        如果二叉树中两个 叶 节点之间的 最短路径长度 小于或者等于 distance ，
        那它们就可以构成一组 好叶子节点对 。返回树中 好叶子节点对的数量 。
        '''
        def dfs(R: TreeNode, t):
            if not R:
                return []
            left, right = [], []
            if R.left and R.right:
                left = dfs(R.left, t + 1)
                right = dfs(R.right, t + 1)
                for l in left:
                    for r in right:
                        if l + r - 2 * t <= distance:
                            self.res += 1
            elif R.left:
                left = dfs(R.left, t + 1)
            elif R.right:
                right = dfs(R.right, t + 1)
            else:
                return [t]
            return left + right

        self.res = 0
        dfs(root, 0)
        return self.res

    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        '''
        desc: 二叉树的锯齿形层次遍历
        给定一个二叉树，返回其节点值的锯齿形层次遍历。
        （即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
        '''
        def dfs(root):
            if not root:
                return []
            stack = [root]
            new_stack = []
            n = len(stack)
            count = 1
            output = []
            tmp = []
            while stack:
                while n>0:
                    node = stack.pop(0)
                    tmp.append(node.val)
                    if node.left:
                        new_stack.append(node.left)
                    if node.right:
                        new_stack.append(node.right)
                    n-=1
                if count%2==1:
                    output.append(tmp)
                else:
                    output.append(tmp[::-1])
                count += 1
                tmp = []
                stack = new_stack
                new_stack = []
                n = len(stack)
            return output
        return dfs(root)

    def countNodes(self, root: TreeNode) -> int:
        '''
        desc: 完全二叉树的节点个数
        给出一个完全二叉树，求出该树的节点个数。
        '''
        def dfs(root):
            if not root:
                return 0
            if not root.left and not root.right:
                return 1
            elif not root.right:
                return 2
            left = dfs(root.left)
            right = dfs(root.right)
            return left+right+1
        return dfs(root)

    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        '''
        desc:  树的子结构
        输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
        B是A的子结构， 即 A中有出现和B相同的结构和节点值。
        '''
        def dfs(A, B):
            if not B:
                return True
            elif not A or A.val!=B.val:
                return False
            else:
                return dfs(A.left, B.left) and dfs(A.right, B.right)
        if not A and not B:
            return True
        if not A or not B:
            return False
        return dfs(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)

    def pruneTree(self, root: TreeNode) -> TreeNode:
        '''
        desc: 二叉树剪枝
        给定二叉树根结点 root ，此外树的每个结点的值要么是 0，要么是 1。
        返回移除了所有不包含 1 的子树的原二叉树。
        ( 节点 X 的子树为 X 本身，以及所有 X 的后代。)
        '''
        def dfs(root):
            if not root:
                return None
            left = dfs(root.left)
            right = dfs(root.right)
            if not left:
                root.left = None
            if not right:
                root.right = None
            return root.val==1 or left or right
            
        return root if dfs(root) else None

    def isValidBST(self, root: TreeNode) -> bool:
        '''
        desc: 合法二叉搜索树
        最简单的解法：先中序遍历 然后判断是否递增
        实现一个函数，检查一棵二叉树是否为二叉搜索树。
        '''
        def dfs(root):
            if not root:
                return True
            left = dfs(root.left)
            if self.pre!=None and pre.val>=root.val:
                return False
            self.pre = root
            right = dfs(root.right)
            return left and right
            
        self.pre = TreeNode(None)
        return dfs(root)

    def pathSum(self, root: TreeNode, sum: int) -> int:
        '''
        desc: 求和路径
        给定一棵二叉树，其中每个节点都含有一个整数数值(该值或正或负)。
        设计一个算法，打印节点数值总和等于某个给定值的所有路径的数量。
        注意，路径不一定非得从二叉树的根节点或叶节点开始或结束，
        但是其方向必须向下(只能从父节点指向子节点方向)。
        '''
        pass




















