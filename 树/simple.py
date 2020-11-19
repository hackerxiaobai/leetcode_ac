from .base import TreeNode, NTreeNode
from typing import List
'''
desc: 树相关简单题
'''
class SimpleTopic(TreeNode):
	"""docstring for SimpleTopic"""
	def preorder(self, root: TreeNode) -> List[int]:
		# 前序二叉树
		if root is None:
			return []
		val = root.value
		left = self.preorder(root.left)
		right = self.preorder(root.right)
		return [val] + left + right

	def midorder(self, root: TreeNode) -> List[int]:
		# 中序二叉树
		if root is None:
			return []
		val = root.value
		left = self.preorder(root.left)
		right = self.preorder(root.right)
		return left + [val] + right

	def endorder(self, root: TreeNode) -> List[int]:
		# 后序二叉树
		if root is None:
			return []
		val = root.value
		left = self.preorder(root.left)
		right = self.preorder(root.right)
		return left + right + [val]

	def n_preorder(self, root: NTreeNode) -> List[int]:
		# N叉树 的 前序遍历
		if root is None:
			return []
		stack = [root]
		output = []
		while stack:
			node = stack.pop()
			output.append(node.value)
			stack.extend(node.children[::-1])
		return output

    def n_postorder(self, root: 'Node') -> List[int]:
        # N叉树 的 后序遍历
        def dfs(root):
            if not root:
                return []
            stack = [root]
            output = []
            while stack:
                node = stack.pop()
                output.append(node.val)
                stack.extend(node.children)
            return output[::-1]
        return dfs(root)    
		
	def isBalanced(self, root: TreeNode) -> bool:
		# 平衡二叉树判定
		def helper(root):
			if root is None:
				return 0
			left = helper(root.left)
			right = helper(root.right)
			if left==-1 or right==-1 or abs(left-right)>1:
				return -1
			else:
				return max(left, right) + 1
		return helper(root)>=0

	def levelOrder(self, root: TreeNode) -> List[List[int]]:
		# 按层输出二叉树，从上到下，从左到右
		if root is None:
			return []
		stack = [root]
		n = len(stack)
		output = []
		tmp = []
		new_stack = []
		while stack:
			while n>0:
				tmp_node = stack.pop(0)
				tmp.append(tmp_node.val)
				if tmp_node.left:
					new_stack.append(tmp_node.left)
				if tmp_node.right:
					new_stack.append(tmp_node.right)
				n -= 1
			stack = new_stack
			n = len(stack)
			output.append(tmp)
			tmp = []
			new_stack = []
		return output

	def increasingBST(self, root: TreeNode) -> TreeNode:
		'''
		desc: 递增顺序查找树
			  给你一个树，请你 按中序遍历 重新排列树，
			  使树中最左边的结点现在是树的根，
			  并且每个结点没有左子结点，只有一个右子结点。
		'''
		def midorder(root):
			# 中序遍历
			if root is None:
				return []
			val = root.val
			left = midorder(root.left)
			right = midorder(root.right)
			return left + [val] + right
        
        cur = head = TreeNode(None)
        for v in midorder(root):
            cur.right = TreeNode(v)
            cur = cur.right
		return head.right

	def kthLargest(self, root: TreeNode, k: int) -> int:
		'''
		desc:  二叉搜索树的第k大节点
				给定一棵二叉搜索树，请找出其中第k大的节点。
		'''
    	def dfs(root):
            if not root: return
            dfs(root.right)
            if self.k == 0: return
            self.k -= 1
            if self.k == 0: self.res = root.val
            dfs(root.left)
		self.k = k
		dfs(root)
		return self.res

	def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    	'''
		desc: 路径总和
			  给定一个二叉树和一个目标和，
			  判断该树中是否存在根节点到叶子节点的路径，
			  这条路径上所有节点值相加等于目标和。
    	'''
    	def dfs(root, target):
    		if root is None:
    			return []

    		if root.left is None and root.right is None and root.val==target:
    			return [[root.val]]
    		ret = []
    		left = dfs(root.left, target-root.val)
    		right = dfs(root.right, target-root.val)
    		for i in left+right:
    			ret.append([root.val]+i)
            return ret
    	return dfs(root, sum)

    def sumRootToLeaf(self, root: TreeNode) -> int:
    	'''
		desc: 从根到叶的二进制数之和
			  给出一棵二叉树，其上每个结点的值都是 0 或 1 。
			  每一条从根到叶的路径都代表一个从最高有效位开始的二进制数。
			  例如，如果路径为 0 -> 1 -> 1 -> 0 -> 1，那么它表示二进制数 01101，也就是 13
    	'''
    	def dfs(root):
            if root is None:
                return []
            if root.left is None and root.right is None:
                return [[str(root.val)]]
            ret = []
            left = dfs(root.left)
            right = dfs(root.right)
            for i in left + right:
                ret.append([str(root.val)]+i)
            return ret

        def binary2ten(nums):
            sum = 0
            for item in nums:
                item = ''.join(item)
                sum += int('0b'+item, 2)
            return sum 
        return binary2ten(dfs(root))

    def tree2str(self, root: TreeNode) -> str:
    	'''
		desc: 根据二叉树创建字符串
			  你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。
			  空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树
			  之间的一对一映射关系的空括号对。
    	'''
    	def dfs(root):
            if root is None:
                return "()"
            if not root.left and not root.right:
                return "("+str(root.val)+")"
            left = dfs(root.left)
            right = dfs(root.right)
            if right=='()':
                right = ''
            return "("+str(root.val)+left+right+")"
        return dfs(t)[1:-1]

    def isSymmetric(self, root: TreeNode) -> bool:
    	'''
		desc: 对称的二叉树
			  请实现一个函数，用来判断一棵二叉树是不是对称的。
			  如果一棵二叉树和它的镜像一样，那么它是对称的。
			  例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    	'''
    	def recur(L, R):
            if not L and not R: return True
            if not L or not R or L.val != R.val: return False
            return recur(L.left, R.right) and recur(L.right, R.left)

        return recur(root.left, root.right) if root else True

    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
    	'''
		desc: 合并二叉树
			  给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，
			  两个二叉树的一些节点便会重叠。
			  你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，
			  那么将他们的值相加作为节点合并后的新值，
			  否则不为 NULL 的节点将直接作为新二叉树的节点。
    	'''
    	if not t1:
            return t2
        if not t2:
            return t1
        
        merged = TreeNode(t1.val + t2.val)
        merged.left = self.mergeTrees(t1.left, t2.left)
        merged.right = self.mergeTrees(t1.right, t2.right)
        return merged

    def maxDepth(self, root: TreeNode) -> int:
    	'''
		desc: 二叉树的最大深度
    	'''
        if root is None:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1
    
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
    	'''
		desc: 二叉树的层次遍历 II
			  给定一个二叉树，返回其节点值自底向上的层次遍历。 
			  （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
    	'''
        if root is None:
            return []
        stack = [root]
        n = len(stack)
        output = []
        tmp = []
        new_stack = []
        while stack:
            while n>0:
                node = stack.pop(0)
                tmp.append(node.val)
                if node.left:
                    new_stack.append(node.left)
                if node.right:
                    new_stack.append(node.right)
                n -= 1
            output.append(tmp)
            tmp = []
            stack = new_stack
            n = len(stack)
            new_stack = []

        return output[::-1]

    def invertTree(self, root: TreeNode) -> TreeNode:
    	'''
		desc: 翻转二叉树
    	'''
        if not root:
            return root
        
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left, root.right = right, left
        return root
    def lowestCommonAncestor(self, root: TreeNode, p: int, q: int) -> TreeNode:
    	'''
		desc: 二叉搜索树的最近公共祖先
    	'''
    	if root is None:
            return root
        if root.val>p.val and root.val>q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val<p.val and root.val<q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

     def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        '''
        desc: 相同的树
        给定两个二叉树，编写一个函数来检验它们是否相同。
        如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
        '''
        def dfs(p, q):
            if not p and not q:
                return True
            if not p and q or p and not q or p.val!=q.val:
                return False
            else:
                return dfs(p.left, q.left) and dfs(p.right, q.right)

        return dfs(p, q)

    def getMinimumDifference(self, root: TreeNode) -> int:
        '''
        desc: 二叉搜索树的最小绝对差
        给你一棵所有节点为非负值的二叉搜索树，
        请你计算树中任意两节点的差的绝对值的最小值。
        '''
        def dfs(root):
            if root is None:
                return 
            dfs(root.left)
            if self.pre == -1:
                self.pre = root.val
            else:
                self.res = min(self.res, abs(root.val-self.pre))
                self.pre = root.val
            dfs(root.right)
            

        self.res = float('inf')
        self.pre = -1
        dfs(root)
        return self.res

    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        '''
        desc: 左叶子之和
        计算给定二叉树的所有左叶子之和。
        '''
        def dfs(root, flag):
            if root is None:
                return 0
            if root.left is None and root.right is None and flag=="left":
                self.res += root.val
            dfs(root.left, "left")
            dfs(root.right, "right")

        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 0
        self.res = 0
        flag = "left"
        dfs(root, flag)
        return self.res

    def isUnivalTree(self, root: TreeNode) -> bool:
        '''
        desc: 单值二叉树
        如果二叉树每个节点都具有相同的值，那么该二叉树就是单值二叉树。
        只有给定的树是单值二叉树时，才返回 true；否则返回 false
        '''
         def dfs(root):
            if root is None:
                return True
            if root.val != self.res:
                return False
            left = dfs(root.left)
            right = dfs(root.right)
            if left!=right:
                return False
            elif not left and not right:
                return False  
            else:
                return True          
        if root is None:
            return True
        self.res = root.val
        return dfs(root)

    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        '''
        desc: 修剪二叉搜索树
        给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。
        通过修剪二叉搜索树，使得所有节点的值在[low, high]中。
        修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，
        原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。
        所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，
        根节点可能会根据给定的边界发生改变。
        '''
        def dfs(root, low, high):
            if root is None:
                return root
            if root.val<low:
                return dfs(root.right, low, high)
            if root.val>high:
                return dfs(root.left, low, high)
            root.left = dfs(root.left, low, high)
            root.right = dfs(root.right, low, high)
            return root
        return dfs(root, low, high)

    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        '''
        desc: 二叉树的堂兄弟节点
        在二叉树中，根节点位于深度 0 处，每个深度为 k 的节点的子节点位于深度 k+1 处。
        如果二叉树的两个节点深度相同，但父节点不同，则它们是一对堂兄弟节点。
        我们给出了具有唯一值的二叉树的根节点 root，以及树中两个不同节点的值 x 和 y。
        只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true。否则，返回 false。
        '''
        def dfs(node, par = None):
            if node:
                depth[node.val] = 1 + depth[par.val] if par else 0
                parent[node.val] = par
                dfs(node.left, node)
                dfs(node.right, node)
        parent = {}
        depth = {}
        dfs(root)
        return depth[x] == depth[y] and parent[x] != parent[y]

    def findTarget(self, root: TreeNode, k: int) -> bool:
        '''
        desc: 两数之和 IV - 输入 BST
        给定一个二叉搜索树和一个目标结果，
        如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。
        '''
        def dfs(root, k, s):
            if root is None:
                return False
            if (k-root.val) in s:
                return True
            s.add(root.val)
            return dfs(root.left, k, s) or dfs(root.right, k, s)
        s = set()
        return dfs(root, k, s)

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        '''
        desc: 二叉树的直径
        给定一棵二叉树，你需要计算它的直径长度。
        一棵二叉树的直径长度是任意两个结点路径长度中的最大值。
        这条路径可能穿过也可能不穿过根结点。
        '''
        
        def dfs(root):
            if not root:
                return 0 
            left = dfs(root.left)
            right = dfs(root.right)
            self.res = max(self.res, left+right+1)
            return max(left, right)+1
        self.res = 0
        dfs(root)
        return self.res -1

    def findSecondMinimumValue(self, root: TreeNode) -> int:
        '''
        desc: 二叉树中第二小的节点
        给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。
        如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。
        更正式地说，root.val = min(root.left.val, root.right.val) 总成立。
        给出这样的一个二叉树，你需要输出所有节点中的第二小的值。
        如果第二小的值不存在的话，输出 -1 。
        '''
        def dfs(root, m):
            if not root:
                return -1
            if root.val>m:
                return root.val
            left = dfs(root.left, m)
            right = dfs(root.right, m)
            if left==-1:
                return right
            if right==-1:
                return left
            return min(left, right)
        if not root:
            return -1
        return dfs(root, root.val)

    def averageOfLevels(self, root: TreeNode) -> List[float]:
        '''
        desc: 二叉树的层平均值
        给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
        '''
        def dfs(root):
            if not root:
                return 
            stack = [root]
            new_stack = []
            output = []
            tmp = 0
            count = n = len(stack)
            while stack:
                while n>0:
                    node = stack.pop(0)
                    tmp += node.val
                    if node.left:
                        new_stack.append(node.left)
                    if node.right:
                        new_stack.append(node.right)
                    n-=1

                output.append(tmp/count)
                stack = new_stack
                new_stack = []
                tmp = 0
                count = n = len(stack)
            return output
        return dfs(root)

    def maxDepth(self, root: 'Node') -> int:
        '''
        desc: N叉树的最大深度
        给定一个 N 叉树，找到其最大深度。
        最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
        '''
        if root is None: 
            return 0 
        elif root.children == []:
            return 1
        else: 
            height = [self.maxDepth(c) for c in root.children]
            return max(height) + 1 

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        '''
        desc: 最小高度树
        给定一个有序整数数组，元素各不相同且按升序排列，
        编写一个算法，创建一棵高度最小的二叉搜索树。
        '''
        def dfs(nums, low, high):
            if low>high:
                return None
            mid = low + (high-low)//2
            node = TreeNode(nums[mid])
            node.left = dfs(nums, low, mid-1)
            node.right = dfs(nums, mid+1, high)
            return node
        return dfs(nums, 0, len(nums)-1)

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        '''
        desc: 二叉搜索树中的搜索
        给定二叉搜索树（BST）的根节点和一个值。
        你需要在BST中找到节点值等于给定值的节点。 
        返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
        '''
        if not root:
            return None
        if root.val>val:
            root = root.left
            return self.searchBST(root, val)
        elif root.val<val:
            root = root.right
            return self.searchBST(root, val)
        else:
            return root

    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        '''
        desc: 二叉搜索树的范围和
        给定二叉搜索树的根结点 root，返回值位于范围 [low, high] 之间的所有结点的值的和。
        '''
        def dfs(node):
            if node:
                if L <= node.val <= R:
                    self.ans += node.val
                if L < node.val:
                    dfs(node.left)
                if node.val < R:
                    dfs(node.right)
        self.ans = 0
        dfs(root)
        return self.ans

    def minDepth(self, root: TreeNode) -> int:
        '''
        desc: 二叉树的最小深度
        给定一个二叉树，找出其最小深度。
        最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
        '''
        def dfs(root):
            if not root:
                return 0
            if not root.left and not root.right:
                return 1
            left = dfs(root.left)
            right = dfs(root.right)
            if not left or not right:
                return left + right + 1
        
            return min(left, right)+1
        
        self.res = float('inf')
        return dfs(root)

    def isBalanced(self, root: TreeNode) -> bool:
        '''
        desc: 检查平衡性
              实现一个函数，检查二叉树是否平衡。
              在这个问题中，平衡树的定义如下：任意一个节点，其两棵子树的高度差不超过 1。
        '''
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            if left==-1 or right==-1 or abs(left-right)>1:
                return -1

            return max(left, right) + 1
        return dfs(root) != -1

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        '''
        desc: 二叉树的所有路径
        给定一个二叉树，返回所有从根节点到叶子节点的路径。
        '''
        def dfs(root):
            if not root:
                return []
            if not root.left and not root.right:
                return [str(root.val)]

            ret = []
            left = dfs(root.left)
            right = dfs(root.right)
            for i in left+right:
                ret.append(str(root.val)+'->'+i)
            return ret
        return dfs(root)

    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        '''
        desc: 叶子相似的树
        请考虑一棵二叉树上所有的叶子，
        这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。
        '''
        def dfs(root):
            if not root:
                return
            if not root.left and not root.right:
                self.l1.append(root.val)
            dfs(root.left)
            dfs(root.right)
        
        self.l1 = []
        dfs(root1)
        self.l2 = self.l1
        self.l1 = []
        dfs(root2)
        return self.l1 == self.l2

    def mirrorTree(self, root: TreeNode) -> TreeNode:
        '''
        desc: 二叉树的镜像
        请完成一个函数，输入一个二叉树，该函数输出它的镜像。
        '''
        def dfs(root):
            if not root:
                return
            tmp = root.left
            root.left = dfs(root.right)
            root.right = dfs(tmp)
            return root
        return dfs(root)

    def convertBiNode(self, root: TreeNode) -> TreeNode:
        '''
        desc: 二叉树数据结构TreeNode可用来表示单向链表
        （其中left置空，right为下一个链表节点）。
        实现一个方法，把二叉搜索树转换为单向链表，
        要求依然符合二叉搜索树的性质，转换操作应是原址的，
        也就是在原始的二叉搜索树上直接修改。返回转换后的单向链表的头节点。
        '''
        def dfs(root):
            if not root:
                return None
            dfs(root.left)
            root.left = None
            self.cur.right = root
            self.cur = root
            dfs(root.right)
        self.ans = self.cur = TreeNode(0)
        dfs(root)
        return self.ans.right






































