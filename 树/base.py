class TreeNode(object):
	"""docstring for Tree"""
	def __init__(self, val):
		super(TreeNode, self).__init__()
		self.val = val
		self.left = None
		self.right = None

class NTreeNode(object):
	"""docstring for NTreeNode"""
	def __init__(self, val):
		super(NTreeNode, self).__init__()
		self.val = val
		self.children = [NTreeNode]
		
		