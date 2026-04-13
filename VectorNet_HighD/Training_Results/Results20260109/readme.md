Train on recording 26-45.





25ep:



(vnpy39) chwei@chwei-System-Product-Name:~/Compared1\_VN\_trajectory\_prediction$ python train\_VN.py

Reading Records: 100%|█████████████████████████████████████████████████████████████████████████████████████| 20/20 \[01:33<00:00,  4.65s/it]

train\_num:valid\_num:test\_num = 317588:90739:45371

/home/chwei/anaconda3/envs/vnpy39/lib/python3.9/site-packages/torch\_geometric/deprecation.py:26: UserWarning: 'data.DataLoader' is deprecat

ed, use 'loader.DataLoader' instead

  warnings.warn(out)

the total number of trajectory: 453698

数据处理运行时间: 93.10 秒

epoch 0: train\_rmse 10.5295 | dev\_rmse 1.5012 m | ADE 1.3129 m | FDE 2.6949 m

epoch 1: train\_rmse 1.2580 | dev\_rmse 1.2729 m | ADE 1.1995 m | FDE 2.5083 m

epoch 2: train\_rmse 1.1473 | dev\_rmse 1.0875 m | ADE 0.9713 m | FDE 2.0998 m

epoch 3: train\_rmse 1.1202 | dev\_rmse 1.1312 m | ADE 1.0305 m | FDE 2.1912 m

epoch 4: train\_rmse 1.0881 | dev\_rmse 1.0775 m | ADE 0.9375 m | FDE 2.0385 m

epoch 5: train\_rmse 0.9749 | dev\_rmse 0.9826 m | ADE 0.8469 m | FDE 1.8890 m

epoch 6: train\_rmse 0.9674 | dev\_rmse 0.9696 m | ADE 0.8366 m | FDE 1.8760 m

epoch 7: train\_rmse 0.9644 | dev\_rmse 0.9545 m | ADE 0.8295 m | FDE 1.8623 m

epoch 8: train\_rmse 0.9577 | dev\_rmse 0.9552 m | ADE 0.8261 m | FDE 1.8617 m

epoch 9: train\_rmse 0.9527 | dev\_rmse 1.0214 m | ADE 0.9040 m | FDE 1.9795 m

epoch 10: train\_rmse 0.9199 | dev\_rmse 0.9206 m | ADE 0.7893 m | FDE 1.7991 m

epoch 11: train\_rmse 0.9178 | dev\_rmse 0.9226 m | ADE 0.7905 m | FDE 1.8017 m

epoch 12: train\_rmse 0.9153 | dev\_rmse 0.9179 m | ADE 0.7829 m | FDE 1.7877 m

epoch 13: train\_rmse 0.9136 | dev\_rmse 0.9250 m | ADE 0.7977 m | FDE 1.8150 m

epoch 14: train\_rmse 0.9133 | dev\_rmse 0.9235 m | ADE 0.7927 m | FDE 1.8025 m

epoch 15: train\_rmse 0.9003 | dev\_rmse 0.9111 m | ADE 0.7809 m | FDE 1.7857 m

epoch 16: train\_rmse 0.8995 | dev\_rmse 0.9123 m | ADE 0.7747 m | FDE 1.7741 m

epoch 17: train\_rmse 0.8990 | dev\_rmse 0.9111 m | ADE 0.7761 m | FDE 1.7768 m

epoch 18: train\_rmse 0.8979 | dev\_rmse 0.9112 m | ADE 0.7804 m | FDE 1.7849 m

epoch 19: train\_rmse 0.8982 | dev\_rmse 0.9107 m | ADE 0.7753 m | FDE 1.7762 m

epoch 20: train\_rmse 0.8938 | dev\_rmse 0.9071 m | ADE 0.7735 m | FDE 1.7728 m

epoch 21: train\_rmse 0.8940 | dev\_rmse 0.9107 m | ADE 0.7801 m | FDE 1.7845 m

epoch 22: train\_rmse 0.8933 | dev\_rmse 0.9093 m | ADE 0.7745 m | FDE 1.7742 m

epoch 23: train\_rmse 0.8931 | dev\_rmse 0.9069 m | ADE 0.7720 m | FDE 1.7700 m

epoch 24: train\_rmse 0.8933 | dev\_rmse 0.9073 m | ADE 0.7733 m | FDE 1.7724 m

训练运行时间: 3094.295621 秒

test loss: 0.9040 m, last train loss: 0.8933, last dev loss: 0.9073 m







ep100:
...
epoch 61: train\_rmse 0.8987 | dev\_rmse 0.9060 m | ADE 0.7638 m | FDE 1.7796 m
epoch 62: train\_rmse 0.8987 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 63: train\_rmse 0.8986 | dev\_rmse 0.9065 m | ADE 0.7639 m | FDE 1.7798 m
epoch 64: train\_rmse 0.8986 | dev\_rmse 0.9062 m | ADE 0.7638 m | FDE 1.7797 m
epoch 65: train\_rmse 0.8987 | dev\_rmse 0.9065 m | ADE 0.7639 m | FDE 1.7798 m
epoch 66: train\_rmse 0.8987 | dev\_rmse 0.9065 m | ADE 0.7638 m | FDE 1.7797 m
epoch 67: train\_rmse 0.8986 | dev\_rmse 0.9058 m | ADE 0.7637 m | FDE 1.7795 m
epoch 68: train\_rmse 0.8987 | dev\_rmse 0.9058 m | ADE 0.7638 m | FDE 1.7795 m
epoch 69: train\_rmse 0.8986 | dev\_rmse 0.9060 m | ADE 0.7638 m | FDE 1.7797 m
epoch 70: train\_rmse 0.8986 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 71: train\_rmse 0.8986 | dev\_rmse 0.9058 m | ADE 0.7638 m | FDE 1.7796 m
epoch 72: train\_rmse 0.8986 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 73: train\_rmse 0.8986 | dev\_rmse 0.9065 m | ADE 0.7639 m | FDE 1.7798 m
epoch 74: train\_rmse 0.8986 | dev\_rmse 0.9062 m | ADE 0.7638 m | FDE 1.7797 m
epoch 75: train\_rmse 0.8986 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7797 m
epoch 76: train\_rmse 0.8986 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 77: train\_rmse 0.8987 | dev\_rmse 0.9061 m | ADE 0.7638 m | FDE 1.7797 m
epoch 78: train\_rmse 0.8988 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 79: train\_rmse 0.8987 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 80: train\_rmse 0.8987 | dev\_rmse 0.9058 m | ADE 0.7638 m | FDE 1.7795 m
epoch 81: train\_rmse 0.8986 | dev\_rmse 0.9058 m | ADE 0.7637 m | FDE 1.7795 m
epoch 82: train\_rmse 0.8986 | dev\_rmse 0.9058 m | ADE 0.7638 m | FDE 1.7796 m
epoch 83: train\_rmse 0.8987 | dev\_rmse 0.9061 m | ADE 0.7638 m | FDE 1.7797 m
epoch 84: train\_rmse 0.8987 | dev\_rmse 0.9062 m | ADE 0.7638 m | FDE 1.7798 m
epoch 85: train\_rmse 0.8987 | dev\_rmse 0.9067 m | ADE 0.7638 m | FDE 1.7797 m
epoch 86: train\_rmse 0.8988 | dev\_rmse 0.9059 m | ADE 0.7638 m | FDE 1.7796 m
epoch 87: train\_rmse 0.8987 | dev\_rmse 0.9065 m | ADE 0.7639 m | FDE 1.7797 m
epoch 88: train\_rmse 0.8986 | dev\_rmse 0.9058 m | ADE 0.7638 m | FDE 1.7796 m
epoch 89: train\_rmse 0.8986 | dev\_rmse 0.9063 m | ADE 0.7639 m | FDE 1.7797 m
epoch 90: train\_rmse 0.8987 | dev\_rmse 0.9061 m | ADE 0.7638 m | FDE 1.7797 m
epoch 91: train\_rmse 0.8986 | dev\_rmse 0.9067 m | ADE 0.7639 m | FDE 1.7798 m
epoch 92: train\_rmse 0.8987 | dev\_rmse 0.9063 m | ADE 0.7638 m | FDE 1.7798 m
epoch 93: train\_rmse 0.8991 | dev\_rmse 0.9058 m | ADE 0.7638 m | FDE 1.7795 m
epoch 94: train\_rmse 0.8987 | dev\_rmse 0.9061 m | ADE 0.7638 m | FDE 1.7797 m
epoch 95: train\_rmse 0.8986 | dev\_rmse 0.9070 m | ADE 0.7639 m | FDE 1.7799 m
epoch 96: train\_rmse 0.8987 | dev\_rmse 0.9068 m | ADE 0.7639 m | FDE 1.7798 m
epoch 97: train\_rmse 0.8987 | dev\_rmse 0.9069 m | ADE 0.7639 m | FDE 1.7798 m
epoch 98: train\_rmse 0.8987 | dev\_rmse 0.9070 m | ADE 0.7639 m | FDE 1.7799 m
epoch 99: train\_rmse 0.8986 | dev\_rmse 0.9062 m | ADE 0.7638 m | FDE 1.7796 m
训练运行时间: 12080.802449 秒
test loss: 0.9044 m, last train loss: 0.8986, last dev loss: 0.9062 m





test_for_de.py:

[rec50, 25ep.pth]
ADE: 0.63 m    FDE: 1.40 m    MDE: 18.23 m 
测试运行时间: 1903.880069 秒

[rec(53,54,55), 100ep.pth]
ADE: 0.66 m    FDE: 1.52 m    MDE: 21.06 m 
测试运行时间: 32.711430 秒