from apo import APO

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=('Ethos', 'Liar', 'Jailbreak', 'Sarcasm'), default='Liar',
                        help='which dataset to use')
    parser.add_argument('--prompt', type=str, default='Determine whether the Statement is a lie (Yes) or not (No) '
                                'based on the Context and other information.', help='this is initial prompt actually')
    parser.add_argument('--num_feedbacks', type=int, default=4)
    parser.add_argument('--steps_per_gradient', type=int, default=6)
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--search_depth', type=int, default=4)
    parser.add_argument('--time_steps', type=int, default=10)
    parser.add_argument('--c', type=float, default=1.0)

    args = parser.parse_args()
    print(args)
    engine = APO(args)
