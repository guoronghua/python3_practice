


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reorderList(head: ListNode) -> None:
    """
    Do not return anything, modify head in-place instead.
    """
    queue = []
    cur = head
    flag = 0
    while cur:
        queue.append(cur)
        cur = cur.next
        flag+=1

    pre = None
    cur = head
    while cur:
        pre = cur.next
        temp = queue.pop()
        if cur == temp:
            break
        temp.next = cur.next
        cur.next = temp
        cur = cur.next.next
    aa = cur
    print(head)

node4 = ListNode(4)
node3 = ListNode(4, next=node4)
node2 = ListNode(4, next=node3)
node1 = ListNode(4, next=node2)
reorderList(node1)