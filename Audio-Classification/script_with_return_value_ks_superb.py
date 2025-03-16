import os
import subprocess
import torch 

error_dict = dict()
error_dict[(0,0)] = 0.6573
error_dict[(0,1)] = 92.031
error_dict[(0,2)] = 84.29
error_dict[(0,3)] = 3.11
error_dict[(0,4)] = 1508
error_dict[(0,5)] = 1613
# error_dict[(0,6)] = 1.09
# error_dict[(0,7)] = 0.25
# error_dict[(0,8)] = 0.79
# error_dict[(0,9)] = 132.43
# error_dict[(0,10)] = 0.8850
# error_dict[(0,11)] = 0.9712
error_dict[(1,0)] = 67.81
error_dict[(1,1)] = 3.0249
error_dict[(1,2)] = 4.23
error_dict[(1,3)] = 4.747
error_dict[(1,4)] = 5.98
error_dict[(1,5)] = 1.612
# error_dict[(1,6)] = 46.679
# error_dict[(1,7)] = 2.7038
# error_dict[(1,8)] = 131.224
# error_dict[(1,9)] = 1.54
# error_dict[(1,10)] = 1.9682
# error_dict[(1,11)] = 7.347
error_dict[(2,0)] = 1.29
error_dict[(2,1)] = 3.228
error_dict[(2,2)] = 0.62
error_dict[(2,3)] = 7.545
error_dict[(2,4)] = 1.39
error_dict[(2,5)] = 2.60
# error_dict[(2,6)] = 0.6382
# error_dict[(2,7)] = 0.7819
# error_dict[(2,8)] = 0.7105
# error_dict[(2,9)] = 0.9647
# error_dict[(2,10)] = 0.944
# error_dict[(2,11)] = 1.011
error_dict[(3,0)] = 2.308
error_dict[(3,1)] = 0.9978
error_dict[(3,2)] = 1.32
error_dict[(3,3)] = 0.554
error_dict[(3,4)] = 2.35
error_dict[(3,5)] = 1.00367
# error_dict[(3,6)] = 1.80
# error_dict[(3,7)] = 2.0717
# error_dict[(3,8)] = 0.79
# error_dict[(3,9)] = 2.03
# error_dict[(3,10)] = 1.24
# error_dict[(3,11)] = 7.215
# error_dict[(4,0)] = 5.55
# error_dict[(4,1)] = 2.10
# error_dict[(4,2)] = 2.21
# error_dict[(4,3)] = 1.906
# error_dict[(4,4)] = 1.10
# error_dict[(4,5)] = 2.86
# # error_dict[(4,6)] = 1.88
# # error_dict[(4,7)] = 105.50
# # error_dict[(4,8)] = 1.74
# # error_dict[(4,9)] = 1.80
# # error_dict[(4,10)] = 0.997
# # error_dict[(4,11)] = 75.78
# error_dict[(5,0)] = 0.987
# error_dict[(5,1)] = 6.64
# error_dict[(5,2)] = 2.26
# error_dict[(5,3)] = 1.36
# error_dict[(5,4)] = 2.322
# error_dict[(5,5)] = 1.46
# # error_dict[(5,6)] = 2.47
# # error_dict[(5,7)] = 2.36
# # error_dict[(5,8)] = 2.834
# # error_dict[(5,9)] = 2.60
# # error_dict[(5,10)] = 2.27
# # error_dict[(5,11)] = 1.56
# error_dict[(6,0)] = 1.756
# error_dict[(6,1)] = 4.883
# error_dict[(6,2)] = 2.769
# error_dict[(6,3)] = 2.732
# error_dict[(6,4)] = 2.88
# error_dict[(6,5)] = 1.77
# # error_dict[(6,6)] = 2.517
# # error_dict[(6,7)] = 1.8945
# # error_dict[(6,8)] = 2.6080
# # error_dict[(6,9)] = 6.479
# # error_dict[(6,10)] = 2.56
# # error_dict[(6,11)] = 1.10
# error_dict[(7,0)] = 2.644
# error_dict[(7,1)] = 2.5055
# error_dict[(7,2)] = 2.16
# error_dict[(7,3)] = 1.80
# error_dict[(7,4)] = 2.23
# error_dict[(7,5)] = 1.225
# # error_dict[(7,6)] = 2.100
# # error_dict[(7,7)] = 1.24
# # error_dict[(7,8)] = 2.808
# # error_dict[(7,9)] = 3.611
# # error_dict[(7,10)] = 1.5978
# # error_dict[(7,11)] = 3.35
# # error_dict[(8,0)] = 2.0179
# # error_dict[(8,1)] = 2.949
# # error_dict[(8,2)] = 63.312
# # error_dict[(8,3)] = 3.341
# # error_dict[(8,4)] = 3.58
# # error_dict[(8,5)] = 1.55
# # error_dict[(8,6)] = 2.972
# # error_dict[(8,7)] = 1.8907
# # error_dict[(8,8)] = 2.827
# # error_dict[(8,9)] = 2.1589
# # error_dict[(8,10)] = 1.6225
# # error_dict[(8,11)] = 2.62
# # error_dict[(9,0)] = 1.39
# # error_dict[(9,1)] = 2.98
# # error_dict[(9,2)] = 2.56
# # error_dict[(9,3)] = 1.87
# # error_dict[(9,4)] = 2.15
# # error_dict[(9,5)] = 1.44
# # error_dict[(9,6)] = 2.12
# # error_dict[(9,7)] = 1.57
# # error_dict[(9,8)] = 1.84
# # error_dict[(9,9)] = 1.0007
# # error_dict[(9,10)] = 1.337
# # error_dict[(9,11)] = 1.73
# # error_dict[(10,0)] = 0.96
# # error_dict[(10,1)] = 0.88
# # error_dict[(10,2)] = 2.5064
# # error_dict[(10,3)] = 2.615
# # error_dict[(10,4)] = 1.856
# # error_dict[(10,5)] = 2.310
# # error_dict[(10,6)] = 1.9255
# # error_dict[(10,7)] = 1.530
# # error_dict[(10,8)] = 1.3736
# # error_dict[(10,9)] = 1.505
# # error_dict[(10,10)] = 0.7426
# # error_dict[(10,11)] = 1.0536
# # error_dict[(11,0)] = 1.219
# # error_dict[(11,1)] = 0.99
# # error_dict[(11,2)] = 2.07
# # error_dict[(11,3)] = 3.2168
# # error_dict[(11,4)] = 0.6232
# # error_dict[(11,5)] = 1.90
# # error_dict[(11,6)] = 1.799
# # error_dict[(11,7)] = 0.8917
# # error_dict[(11,8)] = 0.4054
# # error_dict[(11,9)] = 0.68013
# # error_dict[(11,10)] = 1.255
# # error_dict[(11,11)] = 2.911
# # sorted_error_dict = dict(sorted(error_dict.items(), key=lambda item: item[1]))
# # print(sorted_error_dict)
head_list = dict()
for j in range(12):
    head_list[j] = list()
# x = "/scratch/gilbreth/amohanpa/whisper-tiny-hi/librispeech_clean/"
# x = "/scratch/gilbreth/amohanpa/Whisper-tiny/common_language"
x = "sanchit-gandhi/whisper-tiny-ft-keyword-spotting"
# x = "/scratch/gilbreth/amohanpa/bert-base/sst2/"
# y = "/scratch/gilbreth/amohanpa/bert-base/mrpc/Trail/"
# os.system("python run_glue.py --model_name_or_path {} --task_name mrpc --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 0.187 --num_train_epochs 1 --output_dir {} --overwrite_output_dir --layer0_list '{}' --layer1_list '{}' --layer2_list '{}' --layer3_list '{}' --layer4_list '{}' --layer5_list '{}' --layer6_list '{}' --layer7_list '{}' --layer8_list '{}' --layer9_list '{}' --layer10_list '{}' --layer11_list '{}'".format(x,y,head_list[0],head_list[1],head_list[2],head_list[3],head_list[4],head_list[5],head_list[6],head_list[7],head_list[8],head_list[9],head_list[10],head_list[11]))
# os.system("python run_glue.py --model_name_or_path {} --task_name mrpc --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 0.187 --num_train_epochs 1 --output_dir {} --overwrite_output_dir --layer0_list '{}' --layer1_list '{}' --layer2_list '{}' --layer3_list '{}' --layer4_list '{}' --layer5_list '{}' --layer6_list '{}' --layer7_list '{}' --layer8_list '{}' --layer9_list '{}' --layer10_list '{}' --layer11_list '{}' --layer_num_new {} --head_num_new {}".format(x,y,head_list[0],head_list[1],head_list[2],head_list[3],head_list[4],head_list[5],head_list[6],head_list[7],head_list[8],head_list[9],head_list[10],head_list[11],0,7))
# x = "/scratch/gilbreth/amohanpa/bert-base/mrpc/"
# print(sorted_error_dict)
count = 0
for i in error_dict.keys():
    if True:
        head_list[i[0]].append(i[1])
        print(head_list)
        y = "/scratch/gilbreth/amohanpa/Whisper-tiny/superb_ks/Simple/WithInit/Thresh02/head-by-head-seq_{}_{}_initfixed/".format(i[0],i[1])
        try:
            # result = subprocess.run(["python", "run_glue_no_trainer.py", "--model_name_or_path", x , "--task_name", "mnli", "--seed", "42", "--do_train", "--do_eval", "--max_seq_length", "128","--max_train_samples", "100000", "--per_device_train_batch_size", "32", "--per_device_eval_batch_size", "128", "--gradient_accumulation_steps", "1", "--learning_rate", "0.187", "--num_train_epochs", "1", "--output_dir", y, "--overwrite_output_dir", "--layer0_list", "{}".format(head_list[0]), "--layer1_list","{}".format(head_list[1]),"--layer2_list","{}".format(head_list[2]),"--layer3_list","{}".format(head_list[3]),"--layer4_list","{}".format(head_list[4]),"--layer5_list","{}".format(head_list[5]),"--layer6_list","{}".format(head_list[6]),"--layer7_list","{}".format(head_list[7]),"--layer8_list","{}".format(head_list[8]),"--layer9_list","{}".format(head_list[9]),"--layer10_list","{}".format(head_list[10]),"--layer11_list","{}".format(head_list[11]),"--layer_num_new","{}".format(i[0]),"--head_num_new","{}".format(i[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = subprocess.run(["python", "run_audio_classification.py", "--model_name_or_path", x , "--dataset_name", "superb", "--dataset_config_name", "ks", "--num_train_epochs", "1", "--output_dir", y, "--per_device_train_batch_size", "32", "--gradient_accumulation_steps", "2", "--per_device_eval_batch_size", "64", "--logging_steps", "25", "--learning_rate", "0.187", "--warmup_steps", "0", "--dataloader_num_workers", "1", "--label_column_name", "label", "--do_eval", "--do_train", "--overwrite_output_dir", "--remove_unused_columns", "False", "--attention_mask", "False", "--layer0_list", "{}".format(head_list[0]), "--layer1_list","{}".format(head_list[1]),"--layer2_list","{}".format(head_list[2]),"--layer3_list","{}".format(head_list[3]), "--layer_num_new","{}".format(i[0]),"--head_num_new","{}".format(i[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Result {}".format(result.stdout.decode('utf-8').strip().split('\n')[-1]))
            print(result.stderr.decode('utf-8').strip())
            print(type(result.stdout.decode('utf-8').strip().split('\n')[-1]))
            print(result.stdout.decode('utf-8').strip().split('\n')[-1])
            print(result.stdout.decode('utf-8').strip().split('\n'))
            result_value=float(result.stdout.decode('utf-8').strip().split('\n')[-1]) 
            print(type(result_value))
            if result_value<0.20:
                x = y
            else:
                head_list[i[0]].remove(i[1])
            torch.cuda.empty_cache()
            
        except subprocess.CalledProcessError as e:
            print("Error:",e)
            print("Output:",e.output)
    else:
        break

print(head_list)
print(x)

# try:
#     result = subprocess.run(["python", "run_glue.py", "--model_name_or_path", "/scratch/gilbreth/amohanpa/bert-base/mrpc/", "--task_name", "mrpc", "--do_eval", "--max_seq_length", "128", "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", "32", "--learning_rate", "0.187", "--num_train_epochs", "20", "--output_dir", "./ "], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print("Result {}".format(result.stdout.decode('utf-8').strip().split('\n')[-1]))
# except subprocess.CalledProcessError as e:
#     print("Error:",e)
#     print("Output:",e.output)

