from collections import deque

def dfs_search(graph, start):
    visited = set()
    stack = [start]
    visited.add(start)

    while stack:
        node = stack.pop()
        print(node, end=' ') 
        
        if node in graph:
            for neighbour in reversed(graph[node]):
                if neighbour not in visited:
                    stack.append(neighbour)
                    visited.add(neighbour) 

def bfs_search(graph, start):
    visited = set();
    queue = deque([start])
    visited.add(start);
    
    while queue:
        node = queue.popleft();
        print(node, end=' ')
        
        if node in graph:
            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.add(neighbour)
    

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F']
}
print("--DFS--")
dfs_search(graph, 'A')
print("--BFS--")
bfs_search(graph, 'A')

