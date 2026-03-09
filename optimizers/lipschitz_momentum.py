import torch

class LipschitzMomentum(torch.optim.Optimizer):

    def __init__(self, params, lr=0.001, beta=0.9, lipschitz=1.0):

        defaults = dict(lr=lr, beta=beta, lipschitz=lipschitz)
        super(LipschitzMomentum, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            lr = group["lr"]
            beta = group["beta"]
            L = group["lipschitz"]

            beta = min(beta, 1 / (L + 1e-6))

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["velocity"] = torch.zeros_like(p)

                v = state["velocity"]

                v.mul_(beta).add_(grad)

                p.add_(v, alpha=-lr)

                state["velocity"] = v