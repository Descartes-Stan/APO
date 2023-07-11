import numpy as np
import random
import re
import openai


openai.api_key = "your_API_key"

def LLM(x):
    prompt = x
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt}
            ]
        )
    response = response.choices[0].message.content
    return response

# 对经过gpt-3.5-turbo的代码进行预处理
def process_prediction(LLM_prediction):

    res = 0 if LLM_prediction == 'No' else 1
    return res

def split_dataset(dataset, samples):
    # 生成随机索引
    random_indices = random.sample(range(len(dataset['train'])), samples)
    # 随机选取数据样本
    mini_dataset = dataset['train'].select(random_indices)
    return mini_dataset

def process_errors(errors):

    formatted_errors = [
        f"{error[0]}\nLabel: {'Yes' if error[1] == 1 else 'No'} Prediction: {'Yes' if error[1] == 0 else 'No'}"
        for error in errors]
    print(formatted_errors[0])
    return formatted_errors

class APO:
    def __init__(self, args):

        self.prompt = args.prompt
        self.dataset = args.dataset
        self.num_feedbacks = args.num_feedbacks
        self.steps_per_gradient = args.steps_per_gradient
        self.beam_width = args.beam_width
        self.search_depth = args.search_depth
        self.time_steps = args.time_steps
        self.c = args.c
        self.mini_dataset = split_dataset(self.dataset, 64)
        self.sample_dataset = split_dataset(self.dataset, 50)

    def expand(self, prompt):

        errors = []
        for i in range(len(self.mini_dataset)):
            liar = "Statement: " + self.mini_dataset[i]['text'] + " " + "Context: " + self.mini_dataset[i]['context']
            x = "# Task\n" + prompt + "\n# Output format\nAnswer Yes or No as labels\n# Examples\n" + prompt + "\nStatement: Small businesses (are) going out of business in record numbers. Context: a speech at Liberty University\nLabel: Yes\n# Prediction\nText: " + liar + "\nLabel: "
            y = self.mini_dataset[i]['labels']
            if process_prediction(LLM(x)) != y:
                errors.append((liar, y))
        errors = process_errors(errors)
        random_errors = random.sample(errors, 4)
        # 计算gradients
        gradients = self.LLM_gradients(prompt, random_errors, self.num_feedbacks)
        # 计算p'
        p_skim = self.LLM_edit_prompt(prompt, gradients, random_errors, self.steps_per_gradient)
        # 计算p''
        p_sskim = self.LLM_monte_carlo_successors(p_skim)
        # 返回p'和p''的并集
        return p_skim + p_sskim

    def LLM_gradients(self, prompt, random_errors, num_feedbacks):
        # 模拟计算 LLM 在当前 prompt p 上的梯度
        # 这里简单返回一个随机生成的梯度作为示例
        gradient_list = []
        for i in range(len(random_errors)):
            gradient = LLM(
                "I'm trying to write a zero-shot classifier.\nMy current prompt is:\n" + prompt + "\nBut it gets the following examples wrong:\n" +
                random_errors[i] + "\ngive " + str(
                    num_feedbacks) + " reasons why the prompt could have gotten these examples wrong.")
            gradient_list.append(gradient)
        return gradient_list

    def LLM_edit_prompt(self, prompt, gradients, random_errors, steps_per_gradient):
        # 模拟使用梯度和错误信息编辑当前 prompt p
        # 这里简单返回一个与 p 相同长度的随机编辑结果作为示例
        p_skim = []
        for i in range(len(random_errors)):
            prompt_skim_templates = LLM(
                "I'm trying to write a zero-shot classifier.\nMy current prompt is:\n" + prompt + "\nBut it gets the following examples wrong:\n" +
                random_errors[i] + "\nBased on these examples the problem with this prompt is that " + gradients[
                    i] + "\nBased on the above information, I wrote " + str(
                    steps_per_gradient) + " different improved prompts." + "\nEach prompt is wrapped with <START> and <END>." + "\nThe " + str(
                    steps_per_gradient) + " new prompts are:")
            list_of_strings = [re.sub(r'^\d+\. ', '', s.replace('<START>', '').replace('<END>', '')) for s in
                               re.split(r'\d+\. ', prompt_skim_templates)[1:]]
            for j in range(len(list_of_strings)):
                p_skim.append(list_of_strings[j])
        return p_skim

    def LLM_monte_carlo_successors(self, p_skim):
        # 模拟生成 Monte Carlo 的 prompt 后继
        # 这里简单返回一个带有后缀的 prompt 列表作为示例4
        p_sskim = []
        for i in range(len(p_skim)):
            p_sskim.append(LLM(
                "Generate a variation of the following instruction while keeping the semantic meaning.\n" + p_skim[
                    i]))
        return p_sskim

    def select_with_ucb(self, prompts):
        n = len(prompts)
        Nt = np.zeros(n)
        Qt = np.zeros(n)

        for t in range(1, self.time_steps + 1):
            ucb_values = Qt + self.c * np.sqrt(np.log(t) / Nt)
            pi = np.argmax(ucb_values)  # 选择具有最大 UCB 值的 prompt
            ri_t = self.metric_func(prompts[pi], self.sample_dataset)  # 观测到的奖励
            Nt[pi] += 1
            Qt[pi] += (ri_t - Qt[pi]) / Nt[pi]

        return prompts[np.argmax(Qt)]

    def metric_func(self, prompt, sample_dataset):
        predict_list = []
        label_list = []
        for i in range(len(sample_dataset)):
            liar = "Statement: " + sample_dataset[i]['text'] + " " + "Context: " + sample_dataset[i]['context']
            x = "# Task\n" + prompt + "\n# Output format\nAnswer Yes or No as labels\n# Examples\n" + prompt + "\nStatement: Small businesses (are) going out of business in record numbers. Context: a speech at Liberty University\nLabel: Yes\n# Prediction\nText: " + liar + "\nLabel: "
            y = sample_dataset[i]['labels']
            predict_list.append(process_prediction(LLM(x)))
            label_list.append(y)

        true_positives = sum([1 for true, pred in zip(label_list, predict_list) if true == pred == 1])
        predicted_positives = sum(predict_list)
        actual_positives = sum(label_list)
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return f1_score

    def automatic_prompt_optimization(self):
        B = [self.prompt]
        for i in range(0, self.search_depth):
            C = []
            for p in B:
                C.extend(self.expand(p))
            B = self.select_with_ucb(C)
        p_hat = max(B, key=lambda p: self.metric_func(p, self.sample_dataset))
        return p_hat
        
    def contrast_res(self):
        f1_init = metric_func(self.prompt, self.mini_dataset)
        f1_apo = metric_func(self.p_hat, self.mini_dataset)
    
