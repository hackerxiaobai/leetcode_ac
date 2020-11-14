from base import TreeNode, NTreeNode

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

















