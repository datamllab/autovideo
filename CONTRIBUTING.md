# Contibuting Guide
Contribution to this project is greatly appreciated! If you find any bugs or have any feedback, please create an issue or send a pull request to fix the bug. If you want to contribute codes for new features, please follow this guide. We currently have several plans. Please create an issue or contact us through emails if you have other suggestions.

## Roadmaps
*   **Skeleton-based action recogonition.** We plan to include more action recogonition algorithms such as skeleton-based methods
*   **Object dectection.** We plan to inlcude obeject dectection primitives

## How to Create a Pull Request

If this your first time to contribute to a project, kindly follow the following instructions. You may find [Creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) helpful. Mainly, you need to take the following steps to send a pull request:

*   Click **Fork** in the upper-right corner of the project main page to create a new branch in your local Github.
*   Clone the repo from your local repo in your Github.
*   Make changes in your computer.
*   Commit and push your local changes to your local repo.
*   Send a pull request to merge your local branch to the branches in RLCard project.

## How to Create a New Primitive

AutoVideo uses general pipeline languages. Each module is wrapped as a primitive. Since most of the tasks are supervised, we have provided a supervised base primitive in [autovideo/base/supervised_base.py](autovideo/base/supervised_base.py). Of course, we can also include unsupervised primitives, which are supported in D3M. Please don't hesitate to contact us if you need a guide. To create a supervised primitive, you generally need to follow the steps below:

*   Create a new file and inherit the classes in [autovideo/base/supervised_base.py](autovideo/base/supervised_base.py).
*   Create hyperparameters in `SupervisedHyperparamsBase`.
*   Put the important variables that you want them to be saved and reused into `SupervisedParamsBase`. Be default, we assume you will save the model class.
*   Implement the functions in `SupervisedPrimitiveBase`, such as `fit`, `produce`, etc.
*   Register the primitive in [entry_points.ini](./autovideo/entry_points.ini).
*   Reinstall Autovideo with `pip3 install -e .`.
*   Run `python3 -m d3m index search` and make sure that your primitive is in the list.
*   Then you can use your primitive! You can play with the [pipeline](autovideo/utils/d3m_utils.py#L61) and add your primitive to the pipeline.

Note that there are many more details that are not mentioned in the above steps. You may refer to the example of [TSN](autovideo/recognition/tsn_primitive.py) to learn more.

