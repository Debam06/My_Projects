class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums3 = nums1 + nums2
        nums3.sort()
        if len(nums3)%2 != 0:
            return nums3[len(nums3)//2]
        else:
            return (nums3[len(nums3)//2] + nums3[len(nums3)//2 - 1]) / 2

solution = Solution()

L1 = [1,2]
L2 = [3,4]
result = solution.findMedianSortedArrays(L1,L2)
