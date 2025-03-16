import os
# os.system("cd ./transformers/examples/pytorch/text-classification/")
# os.system("pwd")
head_list_default = {0:[0,4,5,6,8,10],1:[2,3,6,10],2:[1,4,5,9,10],3:[6,8,9,11],4:[1,2,3,5,6,7,8,9,10],5:[0,1,2,3,4,5,6,7,8,9,10,11],6:[0,1,2,3,4,5,6,7,8,9,10,11],7:[0,1,2,3,4,5,6,7,8,9,10,11],8:[0,1,3,4,5,6,7,8,9,11],9:[0,1,2,3,4,5,6,7,8,9,10,11],10:[0,1,2,3,4,5,6,7,8,9,10,11],11:[0,1,2,3,4,5,6,7,8,9,10,11]}
head_list = dict()
for j in head_list_default.keys():
    head_list[j] = list()
x = "/scratch/gilbreth/amohanpa/bert-base/mrpc/head-by-head-seq_0_0/"

for i in head_list_default.keys():
    for k in head_list_default[i]:
        # print(k)
        if (i==0) and (k==0):
            head_list[i].append(k)
            continue
        head_list[i].append(k)
        y = "/scratch/gilbreth/amohanpa/bert-base/mrpc/head-by-head-seq_{}_{}_initfixed/".format(i,k)
        os.system("python run_glue.py --model_name_or_path {} --task_name mrpc --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 0.187 --num_train_epochs 20 --output_dir {} --overwrite_output_dir --layer0_list '{}' --layer1_list '{}' --layer2_list '{}' --layer3_list '{}' --layer4_list '{}' --layer5_list '{}' --layer6_list '{}' --layer7_list '{}' --layer8_list '{}' --layer9_list '{}' --layer10_list '{}' --layer11_list '{}'".format(x,y,head_list[0],head_list[1],head_list[2],head_list[3],head_list[4],head_list[5],head_list[6],head_list[7],head_list[8],head_list[9],head_list[10],head_list[11]))  
        x = y
