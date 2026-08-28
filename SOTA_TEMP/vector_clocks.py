import json
import csv

class VectorClock:
    def __init__(self, node_id, total_nodes):
        self.node_id = node_id
        self.clock = [0] * total_nodes
        
    def tick(self):
        self.clock[self.node_id] += 1
        
    def update(self, remote_clock):
        for i in range(len(self.clock)):
            self.clock[i] = max(self.clock[i], remote_clock[i])
            
    def get_clock(self):
        return list(self.clock)
        
    def __str__(self):
        return str(self.clock)

def is_causal(c1, c2):
    # Returns True if c1 happened before c2
    less_than = False
    for v1, v2 in zip(c1, c2):
        if v1 > v2:
            return False
        if v1 < v2:
            less_than = True
    return less_than

def test_mattern_vector_clocks():
    # 3 Nodes
    n0 = VectorClock(0, 3)
    n1 = VectorClock(1, 3)
    n2 = VectorClock(2, 3)
    
    events = []
    
    # Node 0 does some work
    n0.tick()
    events.append({'event': 'n0_work', 'clock': n0.get_clock()})
    
    # Node 0 sends msg to Node 1
    n0.tick()
    msg_clock = n0.get_clock()
    events.append({'event': 'n0_send_n1', 'clock': msg_clock})
    
    # Node 1 receives
    n1.tick()
    n1.update(msg_clock)
    events.append({'event': 'n1_recv_n0', 'clock': n1.get_clock()})
    
    # Node 2 does independent work
    n2.tick()
    events.append({'event': 'n2_indep_work', 'clock': n2.get_clock()})
    
    with open('E:\\POLYDIM_EINSOF\\SOTA_TEMP\\vector_clocks_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['event', 'clock'])
        writer.writeheader()
        for e in events:
            # Save as string
            e['clock'] = json.dumps(e['clock'])
            writer.writerow(e)
            
    print("Vector Clocks test completed.")
    print("Causal Check (n0_send_n1 -> n1_recv_n0):", is_causal(events[1]['clock'], events[2]['clock']))
    print("Concurrent Check (n0_work vs n2_indep_work):", 
          not is_causal(events[0]['clock'], events[3]['clock']) and 
          not is_causal(events[3]['clock'], events[0]['clock']))

if __name__ == "__main__":
    test_mattern_vector_clocks()
