import argparse
import re
from sys import prefix
from turtle import forward

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('file', type=str, action="store",
                    help='an integer for the accumulator')

def parse_log(file_name, pp_level: int, mp_level: int):
    f = open(file_name, "r")
    forward_pass_cnt = 0
    total_forward_pass = 0
    send_time = None
    comm_overheads = []
    forward_passes = []
    forward_pass_per_rank = []

    for line in f:
        # print(x)
        indexes = [m.start() for m in re.finditer('\[RANK', line)]

        if len(indexes) == 0:
            continue

        for i, index in enumerate(indexes):
            if i == len(indexes) - 1:
                end = len(line) - 1
            else:
                end = indexes[i + 1]
            sub_string = line[index:end]

            rank = int(sub_string[6])
            splitted = sub_string[9:].split(":")
            title = splitted[0].strip()
            # print(splitted)
            if title == "step cmd":
                pass
            elif title == "send time":
                assert send_time == None
                value = splitted[1].strip()
                send_time = float(value)
            elif title == "received time":
                assert send_time != None
                value = splitted[1].strip()
                try:
                    recv_time = float(value)
                except:
                    value = value.split("[")[0]
                    recv_time = float(value)
                comm_overheads.append(recv_time - send_time)
                send_time = None
            elif title == "forward pass elapsed":
                latency = float(splitted[1].strip())
                assert not rank in forward_pass_per_rank  
                forward_pass_per_rank.append(latency)
                forward_pass_cnt += 1
                
                if pp_level > 1:
                    assert forward_pass_cnt - 1 == rank
                    if rank == pp_level -1:
                        total_forward_pass = sum(forward_pass_per_rank)
                        forward_passes.append(total_forward_pass)
                        forward_pass_cnt = 0
                        forward_pass_per_rank.clear()      
                elif mp_level > 1:
                    if forward_pass_cnt == mp_level:
                        forward_pass = max(forward_pass_per_rank)
                        forward_passes.append(forward_pass)
                        forward_pass_cnt = 0
                        forward_pass_per_rank.clear()
                else:
                    forward_passes.append(latency)
    # remove warmup
    
    print("file name: ", file_name)
    
    
    forward_passes.pop(0)
    avg_fwd_pass_in_ms = sum(forward_passes) / len(forward_passes)
    print("\tavg_fwd_pass: {:.3f}ms".format(avg_fwd_pass_in_ms))

    if len(comm_overheads) > 0:
        comm_overheads.pop(0)
        avg_comm_overhead_in_ms = sum(comm_overheads) / len(comm_overheads) * 1000
        print("\tavg comm overhead: {:.3f}ms".format(avg_comm_overhead_in_ms))
    else:
        avg_comm_overhead_in_ms = 0

    print("\tavg total latency: {:.3f}ms".format(avg_comm_overhead_in_ms + avg_fwd_pass_in_ms))

    # print("""file name: {},
    #     avg_fwd_pass: {:.3f}ms,
    #     gpu bandwidth: {:.3f}GB/s,
    #     comm_overhead proportion: {:.3f}%"""
    #       .format(
    #           file_name,
    #           avg_comm_overhead_in_ms,
    #           ,
    #           avg_comm_overhead_in_ms + avg_fwd_pass_in_ms,
    #           (moving_tensor_size_in_MB / 1024) /
    #           (avg_comm_overhead_in_ms / 1000),
    #           avg_comm_overhead_in_ms /
    #           (avg_fwd_pass_in_ms + avg_comm_overhead_in_ms) * 100)
    #       )


if __name__ == "__main__":
    args = parser.parse_args()

    prefix = 'pp2-g2-b'

    prefix = "dense-g1-b"
    prefix = "mp2-g2-b" 
    if prefix.split("-")[0].startswith("pp"):
        pp_level = int(prefix.split("-")[0][-1])
    else:
        pp_level = 1
    
    if prefix.split("-")[0].startswith("mp"):
        mp_level = int(prefix.split("-")[0][-1])
    else:
        mp_level = 1

    file_names = [f"./{prefix}{i}.txt" for i in [1,2,4,8,16]]
    for file_name in file_names:
        parse_log(file_name, pp_level=pp_level, mp_level=mp_level)
    # argparse.
