import os
import subprocess

error_dict = dict()
error_dict[(0,0)] = 2.34
error_dict[(0,1)] = 7.40
error_dict[(0,2)] = 2.57

error_dict[(1,0)] = 7.36
error_dict[(1,1)] = 2.49
error_dict[(1,2)] = 2.37

error_dict[(2,0)] = 7.39
error_dict[(2,1)] = 7.42
error_dict[(2,2)] = 3.76

error_dict[(3,0)] = 2.61
error_dict[(3,1)] = 3.27
error_dict[(3,2)] = 2.39

error_dict[(4,0)] = 3.56
error_dict[(4,1)] = 2.40
error_dict[(4,2)] = 3.97

error_dict[(5,0)] = 2.26
error_dict[(5,1)] = 7.34
error_dict[(5,2)] = 2.30

error_dict[(6,0)] = 2.25
error_dict[(6,1)] = 2.32
error_dict[(6,2)] = 2.31

error_dict[(7,0)] = 2.27
error_dict[(7,1)] = 2.28
error_dict[(7,2)] = 2.28

error_dict[(8,0)] = 2.29
error_dict[(8,1)] = 2.28
error_dict[(8,2)] = 5.72

error_dict[(9,0)] = 2.27
error_dict[(9,1)] = 2.27
error_dict[(9,2)] = 2.44

error_dict[(10,0)] = 2.25
error_dict[(10,1)] = 2.24
error_dict[(10,2)] = 2.24

error_dict[(11,0)] = 2.25
error_dict[(11,1)] = 2.23
error_dict[(11,2)] = 2.23

sorted_error_dict = dict(sorted(error_dict.items(), key=lambda item: item[1]))
# print(sorted_error_dict)
head_list = dict()
for j in range(12):
    head_list[j] = list()
x = "deit_tiny_patch16_224.fb_in1k"
# y = "/scratch/gilbreth/amohanpa/bert-base/mrpc/Trail/"
# os.system("python run_glue.py --model_name_or_path {} --task_name mrpc --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 0.187 --num_train_epochs 1 --output_dir {} --overwrite_output_dir --layer0_list '{}' --layer1_list '{}' --layer2_list '{}' --layer3_list '{}' --layer4_list '{}' --layer5_list '{}' --layer6_list '{}' --layer7_list '{}' --layer8_list '{}' --layer9_list '{}' --layer10_list '{}' --layer11_list '{}'".format(x,y,head_list[0],head_list[1],head_list[2],head_list[3],head_list[4],head_list[5],head_list[6],head_list[7],head_list[8],head_list[9],head_list[10],head_list[11]))
# os.system("python run_glue.py --model_name_or_path {} --task_name mrpc --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 0.187 --num_train_epochs 1 --output_dir {} --overwrite_output_dir --layer0_list '{}' --layer1_list '{}' --layer2_list '{}' --layer3_list '{}' --layer4_list '{}' --layer5_list '{}' --layer6_list '{}' --layer7_list '{}' --layer8_list '{}' --layer9_list '{}' --layer10_list '{}' --layer11_list '{}' --layer_num_new {} --head_num_new {}".format(x,y,head_list[0],head_list[1],head_list[2],head_list[3],head_list[4],head_list[5],head_list[6],head_list[7],head_list[8],head_list[9],head_list[10],head_list[11],0,7))
# x = "/scratch/gilbreth/amohanpa/bert-base/mrpc/"
# print(sorted_error_dict)
# python train.py --data-dir /depot/araghu/data/Datasets/imagenet2012/ --train-split train --val-split val --batch-size 128 --lr 0.187 --warmup-epochs 0 --model x --pretrained --output y --epochs 1 --layer_num_approx $l --head_num_approx $h
count = 0
print("In script")
for i in sorted_error_dict.keys():

    head_list[i[0]].append(i[1])
    print(head_list)
    if count == 0:
        # y = "/scratch/gilbreth/amohanpa/DeiT/16ChannelWithSkipNew/head-by-head-seq_{}_{}_initfixed/".format(i[0],i[1])
        y = "/scratch/gilbreth/amohanpa/DeiT/Simple3/Cifar100/WithInit/2Epochs/head-by-head-seq_{}_{}_initfixed/".format(i[0],i[1])
        try:
            # result = subprocess.run(["python", "train.py", "--model", "deit_tiny_patch16_224.fb_in1k" , "--data-dir", "/depot/araghu/data/Datasets/imagenet2012/", "--experiment", "seq_train", "--train-split", "train", "--val-split", "val", "--batch-size", "128","--lr", "0.187", "--pretrained", "--warmup-epochs", "0", "--output", y, "--epochs", "1", "--layer0_list", "{}".format(head_list[0]), "--layer1_list","{}".format(head_list[1]),"--layer2_list","{}".format(head_list[2]),"--layer3_list","{}".format(head_list[3]),"--layer4_list","{}".format(head_list[4]),"--layer5_list","{}".format(head_list[5]),"--layer6_list","{}".format(head_list[6]),"--layer7_list","{}".format(head_list[7]),"--layer8_list","{}".format(head_list[8]),"--layer9_list","{}".format(head_list[9]),"--layer10_list","{}".format(head_list[10]),"--layer11_list","{}".format(head_list[11]),"--layer_num_new","{}".format(i[0]),"--head_num_new","{}".format(i[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = subprocess.run(["python", "train.py", "--model", "deit_tiny_patch16_224.fb_in1k" , "--dataset", "torch/cifar100", "--data-dir", "/scratch/gilbreth/amohanpa/Cifar100/", "--experiment", "seq_train", "--train-split", "train", "--val-split", "test", "--batch-size", "128","--lr", "0.187", "--pretrained", "--warmup-epochs", "0", "--output", y, "--epochs", "2", "--layer0_list", "{}".format(head_list[0]), "--layer1_list","{}".format(head_list[1]),"--layer2_list","{}".format(head_list[2]),"--layer3_list","{}".format(head_list[3]),"--layer4_list","{}".format(head_list[4]),"--layer5_list","{}".format(head_list[5]),"--layer6_list","{}".format(head_list[6]),"--layer7_list","{}".format(head_list[7]),"--layer8_list","{}".format(head_list[8]),"--layer9_list","{}".format(head_list[9]),"--layer10_list","{}".format(head_list[10]),"--layer11_list","{}".format(head_list[11]),"--layer_num_new","{}".format(i[0]),"--head_num_new","{}".format(i[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Result {}".format(result.stdout.decode('utf-8').strip().split('\n')[-1]))
            print(result.stderr.decode('utf-8').strip())
            print(type(result.stdout.decode('utf-8').strip().split('\n')[-1]))
            print(result.stdout.decode('utf-8').strip().split('\n')[-1])
            print(result.stdout.decode('utf-8').strip().split('\n'))
            result_value=float(result.stdout.decode('utf-8').strip().split('\n')[-1]) 
            print(type(result_value))
            if result_value<2.4:
                x = y+"seq_train/model_best.pth.tar"
                count=1
            else:
                head_list[i[0]].remove(i[1])
            
        except subprocess.CalledProcessError as e:
            print("Error:",e)
            print("Output:",e.output)
    else:
        # y = "/scratch/gilbreth/amohanpa/DeiT/16ChannelWithSkipNew/head-by-head-seq_{}_{}_initfixed/".format(i[0],i[1])
        y = "/scratch/gilbreth/amohanpa/DeiT/Simple3/Cifar100/WithInit/2Epochs/head-by-head-seq_{}_{}_initfixed/".format(i[0],i[1])
        try:
            result = subprocess.run(["python", "train.py", "--model", "deit_tiny_patch16_224.fb_in1k" , "--initial-checkpoint", x, "--dataset", "torch/cifar100", "--data-dir", "/scratch/gilbreth/amohanpa/Cifar100/", "--experiment", "seq_train", "--train-split", "train", "--val-split", "val", "--batch-size", "128","--lr", "0.187", "--pretrained", "--warmup-epochs", "0", "--output", y, "--epochs", "2", "--layer0_list", "{}".format(head_list[0]), "--layer1_list","{}".format(head_list[1]),"--layer2_list","{}".format(head_list[2]),"--layer3_list","{}".format(head_list[3]),"--layer4_list","{}".format(head_list[4]),"--layer5_list","{}".format(head_list[5]),"--layer6_list","{}".format(head_list[6]),"--layer7_list","{}".format(head_list[7]),"--layer8_list","{}".format(head_list[8]),"--layer9_list","{}".format(head_list[9]),"--layer10_list","{}".format(head_list[10]),"--layer11_list","{}".format(head_list[11]),"--layer_num_new","{}".format(i[0]),"--head_num_new","{}".format(i[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Result {}".format(result.stdout.decode('utf-8').strip().split('\n')[-1]))
            print(result.stderr.decode('utf-8').strip())
            print(type(result.stdout.decode('utf-8').strip().split('\n')[-1]))
            print(result.stdout.decode('utf-8').strip().split('\n')[-1])
            print(result.stdout.decode('utf-8').strip().split('\n'))
            result_value=float(result.stdout.decode('utf-8').strip().split('\n')[-1]) 
            print(type(result_value))
            if result_value<2.4:
                x = y+"seq_train/model_best.pth.tar"
            else:
                head_list[i[0]].remove(i[1])

        except subprocess.CalledProcessError as e:
            print("Error:",e)
            print("Output:",e.output)  

print(head_list)
print(x)

# try:
#     result = subprocess.run(["python", "run_glue.py", "--model_name_or_path", "/scratch/gilbreth/amohanpa/bert-base/mrpc/", "--task_name", "mrpc", "--do_eval", "--max_seq_length", "128", "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", "32", "--learning_rate", "0.187", "--num_train_epochs", "20", "--output_dir", "./ "], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print("Result {}".format(result.stdout.decode('utf-8').strip().split('\n')[-1]))
# except subprocess.CalledProcessError as e:
#     print("Error:",e)
#     print("Output:",e.output)

