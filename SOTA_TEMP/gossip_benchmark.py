import numpy as np
import time
import csv
import sys

def simulate_gossip(num_nodes=100, fanout=3, dim=10000, max_rounds=50):
    # Node states: 0 (ignorant), 1 (infected/has tensor)
    state = np.zeros(num_nodes, dtype=np.int8)
    state[0] = 1 # Seed node
    
    # Tensor size in bytes (Float32 = 4 bytes)
    tensor_size_bytes = dim * 4
    
    history = []
    total_messages = 0
    collisions = 0
    
    start_time = time.time()
    
    for r in range(max_rounds):
        infected_nodes = np.where(state == 1)[0]
        coverage = len(infected_nodes) / num_nodes
        
        history.append({
            'round': r,
            'coverage': coverage,
            'messages_sent': total_messages,
            'collisions': collisions
        })
        
        if coverage == 1.0:
            break
            
        # Each infected node picks `fanout` random targets (excluding itself)
        new_infections = np.zeros(num_nodes, dtype=np.int8)
        receivers_this_round = {}
        
        for node in infected_nodes:
            targets = np.random.choice(num_nodes, fanout, replace=False)
            for t in targets:
                total_messages += 1
                if t in receivers_this_round:
                    receivers_this_round[t] += 1
                else:
                    receivers_this_round[t] = 1
                
                if state[t] == 0:
                    new_infections[t] = 1
                    
        # Calculate collisions (nodes receiving more than 1 message this round)
        for t, count in receivers_this_round.items():
            if count > 1:
                collisions += (count - 1)
                
        state = np.logical_or(state, new_infections).astype(np.int8)
        
    end_time = time.time()
    
    with open('E:\\POLYDIM_EINSOF\\SOTA_TEMP\\gossip_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'coverage', 'messages_sent', 'collisions'])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
            
    print(f"Gossip Benchmark Completed in {end_time - start_time:.4f}s")
    print(f"Nodes: {num_nodes}, Fanout: {fanout}, D: {dim}")
    print(f"Final Coverage: {coverage*100}% in {len(history)} rounds.")
    print(f"Total Messages: {total_messages}, Collisions: {collisions}")
    print(f"Total Bandwidth Used: {(total_messages * tensor_size_bytes) / (1024**2):.2f} MB")
    
if __name__ == "__main__":
    np.random.seed(42)
    # Stress test
    simulate_gossip(num_nodes=1000, fanout=3, dim=50000, max_rounds=100)
