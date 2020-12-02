from base import ListNode

class Solution(ListNode):
	def threeSumClosest(self, nums: List[int], target: int) -> int:
		'''
		desc: 最接近的三数之和
		给定一个包括 n 个整数的数组 nums 和 一个目标值 target。
		找出 nums 中的三个整数，使得它们的和与 target 最接近。
		返回这三个数的和。假定每组输入只存在唯一答案。
		'''
		nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(len(nums)):
            left = i+1
            right = len(nums)-1
            while left<right:
                sum = nums[left] + nums[right] + nums[i]
                if abs(target-sum) < abs(target-ans):
                    ans = sum
                if sum > target:
                    right -= 1
                elif sum<target:
                    left += 1
                else:
                    return ans
        return ans