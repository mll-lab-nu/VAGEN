import heapq
from collections import defaultdict
from typing import List, Set

class TopKStringTracker:
    """
    Efficient Top-K string tracking data structure
    
    Core ideas:
    1. Use hash table to record string counts
    2. Use min-heap to maintain top-m strings (heap root is the m-th largest element)
    3. Lazy cleanup: avoid frequent heap operations
    """
    
    def __init__(self, m: int):
        """
        Initialize the data structure
        
        Args:
            m: Maximum number of strings to retain
        """
        self.m = m
        self.count = defaultdict(int)  # string -> occurrence count
        self.heap = []  # min-heap: (count, string)
        self.in_heap = set()  # track strings in heap to avoid duplicates
        
    def add_strings(self, strings: List[str]) -> None:
        """
        Add k strings to the data structure
        
        Args:
            strings: List of strings to add
        """
        # Convert string list to count dictionary
        string_counts = defaultdict(int)
        for s in strings:
            string_counts[s] += 1
        
        # Use add_string_dict to handle the actual addition
        self.add_string_dict(dict(string_counts))
    
    def add_string_dict(self, string_counts: dict) -> None:
        """
        Add strings with their counts from a dictionary
        
        Args:
            string_counts: Dictionary mapping strings to their occurrence counts
        """
        # Update counts from dictionary
        for string, count in string_counts.items():
            if count <= 0:  # Skip invalid counts
                continue
                
            self.count[string] += count
            
            # If string is already in heap, we don't immediately update heap
            # Instead, we perform lazy update when needed
            if string in self.in_heap:
                continue
                
            # If heap is not full, add directly
            if len(self.heap) < self.m:
                heapq.heappush(self.heap, (self.count[string], string))
                self.in_heap.add(string)
            else:
                # Heap is full, check if we need to replace the heap root
                min_count, min_string = self.heap[0]
                if self.count[string] > min_count:
                    # Remove heap root
                    heapq.heappop(self.heap)
                    self.in_heap.remove(min_string)
                    # Add new element
                    heapq.heappush(self.heap, (self.count[string], string))
                    self.in_heap.add(string)
        
        # Cleanup and rebuild heap (handle count update cases)
        self._cleanup_heap()
    
    def _cleanup_heap(self) -> None:
        """
        Clean up outdated counts in heap and rebuild heap structure
        """
        # Collect current counts of all strings in heap
        current_items = []
        for count, string in self.heap:
            if string in self.count:  # String still exists
                current_items.append((self.count[string], string))
        
        # Rebuild heap
        self.heap = []
        self.in_heap = set()
        
        # Sort by count and take top m
        current_items.sort(reverse=True)
        for count, string in current_items[:self.m]:
            heapq.heappush(self.heap, (count, string))
            self.in_heap.add(string)
    
    def get_top_k(self, k: int) -> Set[str]:
        """
        Return the set of top-k strings by occurrence count
        
        Args:
            k: Number of strings to return
            
        Returns:
            Set containing the top-k strings
        """
        # Get all strings with their counts
        all_items = [(count, string) for string, count in self.count.items()]
        
        # Sort and take top k
        all_items.sort(reverse=True, key=lambda x: x[0])
        
        return {string for _, string in all_items[:k]}
    
    def trim_to_m(self) -> None:
        """
        Keep only the top-m strings by occurrence count, delete others
        """
        if len(self.count) <= self.m:
            return
        
        # Get all strings with their counts, sort by count
        all_items = [(count, string) for string, count in self.count.items()]
        all_items.sort(reverse=True, key=lambda x: x[0])
        
        # Keep top m
        top_m_strings = {string for _, string in all_items[:self.m]}
        
        # Update count dictionary
        self.count = {s: self.count[s] for s in top_m_strings}
        
        # Rebuild heap
        self.heap = [(self.count[s], s) for s in top_m_strings]
        heapq.heapify(self.heap)
        self.in_heap = set(top_m_strings)
    
    def size(self) -> int:
        """Return the number of strings currently stored"""
        return len(self.count)
    
    def get_count(self, string: str) -> int:
        """Get the occurrence count of a specific string"""
        return self.count.get(string, 0)


# Usage example and test code
def test_topk_tracker():
    """Test the functionality of TopKStringTracker"""
    print("=== TopK String Tracker Test ===")
    
    # Initialize, keep top 5 strings
    tracker = TopKStringTracker(m=5)
    
    # First batch of strings
    print("\n1. Adding first batch of strings:")
    batch1 = ["apple", "banana", "apple", "cherry"]
    tracker.add_strings(batch1)
    print(f"Added: {batch1}")
    print(f"Current top-3: {tracker.get_top_k(3)}")
    print(f"Current storage size: {tracker.size()}")
    
    # Second batch of strings
    print("\n2. Adding second batch of strings:")
    batch2 = ["banana", "banana", "date", "elderberry"]
    tracker.add_strings(batch2)
    print(f"Added: {batch2}")
    print(f"Current top-3: {tracker.get_top_k(3)}")
    print(f"String counts:")
    for s in ["apple", "banana", "cherry", "date", "elderberry"]:
        print(f"  {s}: {tracker.get_count(s)}")
    
    # Third batch with more strings
    print("\n3. Adding third batch of strings:")
    batch3 = ["fig", "grape", "apple", "banana", "fig", "grape"]
    tracker.add_strings(batch3)
    print(f"Added: {batch3}")
    print(f"Current top-4: {tracker.get_top_k(4)}")
    print(f"Current storage size: {tracker.size()}")
    
    # Test adding strings from dictionary
    print("\n4. Adding strings from dictionary:")
    string_dict = {"apple": 2, "kiwi": 5, "mango": 3, "banana": 1}
    tracker.add_string_dict(string_dict)
    print(f"Added dict: {string_dict}")
    print(f"Current top-5: {tracker.get_top_k(5)}")
    print(f"String counts after dict addition:")
    for s in ["apple", "banana", "kiwi", "mango", "fig", "grape"]:
        print(f"  {s}: {tracker.get_count(s)}")
    
    # Trim to top 5
    print("\n5. Trimming to top 5 strings:")
    tracker.trim_to_m()
    print(f"Storage size after trimming: {tracker.size()}")
    print(f"Top-5 after trimming: {tracker.get_top_k(5)}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_topk_tracker()