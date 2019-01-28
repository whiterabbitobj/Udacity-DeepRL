[33mcommit 00f8750a02ae17e7673b1a7480b7137b4be34792[m[33m ([m[1;36mHEAD -> [m[1;32mmaster[m[33m)[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Sat Jan 26 18:58:47 2019 -0600

    tuned some stuff. prepping to add DDQN, I think there is a bug in the udacity implementation of DQN around how often the network is updated as well as how the mse_loss is calculate (backwards perhaps?) this is a commit before branching to test new versions

[33mcommit 332cc6fe963c3fbe037771ff3f2eec8031c6d58d[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Sat Jan 26 16:42:29 2019 -0600

    Save/Load matured and bugtested. SGD optimization added. Moving on to new agent types.

[33mcommit 78e7a8654d112ff9f4afa7cf6a00fc71c097ad31[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Sat Jan 26 14:38:24 2019 -0600

    yet more reorganizing, and committing prior to another big code restructure to allow for testing instead of training agent

[33mcommit 70ee972bf73ffa068da3c4b667019498fc3fbf8a[m[33m ([m[1;31morigin/master[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Sat Jan 26 00:17:49 2019 -0600

    End of day checkin. TODO: additional agent types, clean up order of operations, implement previewing the trained agent after loading a checkpoint.

[33mcommit f5f21fc279816d33bba43b6a8b535932511462c1[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Fri Jan 25 23:57:16 2019 -0600

    finished load checkpoint function, but having trouble with reloading the Agent class

[33mcommit b523c60c205cd312df3ae1af41f2f44e56f86fbd[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Fri Jan 25 15:25:40 2019 -0600

    first stage of redesign done, about to start paring back saved/loaded params to see what is actually needed

[33mcommit a0209c2c1287ca40a92573dbe2276fea33dac14e[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Fri Jan 25 14:41:41 2019 -0600

    added save checkpoint functionality and about to begin a major architecture rework in order to pave the road for multiple agent types and flexible network building

[33mcommit 297d251358aa043867d01082fce8c7f037a5428e[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Fri Jan 25 12:19:32 2019 -0600

    Changed overall script layout and some arg names

[33mcommit 2919e3efb56e86334e65c22c65cac8542efd71fe[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Fri Jan 25 11:52:46 2019 -0600

    Added additional print information and cleaned up code a bit.

[33mcommit b77ea3918521721d8729cbc1bd87129c6dd70a25[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Fri Jan 25 11:04:18 2019 -0600

    Code is running. Reorganized some file structure for GITHub presentation moving forward

[33mcommit 2b97f7fb7d27c6ff95dfc4a433870452b29cac96[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 24 19:50:55 2019 -0600

    Squashed more random bugs and adjusted print statements. This is a commit after solving a package binary incompatibility with CUDA versions and my GPU

[33mcommit 780200eef3af4702941a3c9b46bdd1bf1e800fb9[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 24 17:39:06 2019 -0600

    Fixed a bunch of really dumb bugs unrelated to the DRL. Moving on to why the script hangs when the Agent is enabled.

[33mcommit 14a3d7a54f06706a59eab4f2fde1106455b82156[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 24 17:14:54 2019 -0600

    All code implemented and read through carefully for bugs... only thing left to do is try a first run!

[33mcommit c7d0c247a7f5da791146284580a91d5fd2a7cedf[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 24 15:45:37 2019 -0600

    completed first pass at banana_agent main(), now working on implementing the agent code

[33mcommit 96b7a49aee4e0c3553eac0828f9937903c1e03cc[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 24 13:59:51 2019 -0600

    Getting started writing the code/putting together all the pieces

[33mcommit d5993da9deb5fc32eae4b139efab7818a29ac487[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Tue Jan 22 15:59:43 2019 -0600

    add all latest local files

[33mcommit 2ba2536410e358607bf6c2e660f36e31b6a58339[m
Merge: dc65050 8df3f41
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Tue Jan 22 15:42:45 2019 -0600

    Merge branch 'master' of https://github.com/whiterabbitobj/deep-reinforcement-learning

[33mcommit dc65050c8f47b365560a30a112fb84f762005c6b[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:58:45 2019 -0800

    Add files via upload

[33mcommit 855e8c2588294ddd3e1f3a9cb328215747caace3[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:58:02 2019 -0800

    Delete foo.txt

[33mcommit 4460ac6b00a34d498ed6787f7ca6e97926254530[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:57:45 2019 -0800

    Add files via upload

[33mcommit 855e1666331a80d9a0ec0de5cb6881d10fb4b160[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:57:24 2019 -0800

    Create foo.txt

[33mcommit 98ad2b90d89fa5669c319f2c8d205d9057cc492f[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:56:46 2019 -0800

    Add files via upload

[33mcommit a9fd3c62e3bd309d01089b30e7d946ff65a14cd8[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:55:58 2019 -0800

    Delete foo.txt

[33mcommit dc5e073d7042a4fbe2576f8ee100a020ed7dae5b[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:55:39 2019 -0800

    Add files via upload

[33mcommit b8ba712fce5d79461e7f611cb53dff4c19760f6c[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:53:12 2019 -0800

    Create foo.txt

[33mcommit 296a109691068f09147b8db850261499298aea4a[m
Author: juanudacity <35816618+juanudacity@users.noreply.github.com>
Date:   Fri Jan 18 17:52:41 2019 -0800

    Update README.md

[33mcommit 8df3f414e0dc0e655d351219f14903a4d68a6657[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 17 16:44:13 2019 -0600

    Q_Learning implemented for Taxi-v2. Would like to look into MAXQ as a follow up task

[33mcommit 65976d53ef30e9acd779fad4bb717a640d882157[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Thu Jan 17 16:20:24 2019 -0600

    All work implemented, but getting -inf values for reward returns.

[33mcommit 6d771b38c0238423e74d3695bf7e47183d4e3674[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Tue Jan 15 17:27:20 2019 -0600

    quick update for testing fidelity of running locally

[33mcommit a195db6aa8d3504ab38270e5724678088ef17e2a[m
Author: Matthew Doll <mattdoll@gmail.com>
Date:   Sun Jan 13 10:47:56 2019 -0600

    initial commit test

[33mcommit d6cb43c1b11b1d55c13ac86d6002137c7b880c15[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Fri Aug 31 16:54:30 2018 -0500

    ddpg-bipedal compatible with torch0.4

[33mcommit 5e5d32f8ae9857b229d242d5e85037198ddd4ad9[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Tue Aug 28 10:40:33 2018 -0500

    adding project 3

[33mcommit cbebcf325387901b1e5157340cbd7cdeff473096[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Tue Aug 28 09:56:17 2018 -0500

    fix description of crawler

[33mcommit a8cb50c6ecf065983c88d6b96d5621c06c9c20aa[m
Author: Alexis Cook <alexis.cook@acook-mbp.local>
Date:   Wed Aug 8 12:14:45 2018 -0500

    add project 2

[33mcommit b23879aad656b653753c95213ebf1ac111c1d2a6[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Tue Jul 17 10:15:17 2018 -0500

    make compatible with matplotlib==2.2.2

[33mcommit 062b0a421663e90ef8171c6da849c52e4b16623d[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Sat Jul 7 10:22:22 2018 -0500

    Update README.md

[33mcommit f1644944116f36104588aa2932ee224cce0209b5[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Sat Jul 7 10:13:23 2018 -0500

    Update README.md

[33mcommit 55474449a112fa72323f484c4b7a498c8dc84be1[m
Author: Alexis Cook <alexis.cook@acook-mbp.local>
Date:   Fri Jul 6 13:42:07 2018 -0500

    add all files

[33mcommit ff738bfb339a70ec11232f981b0fbf96ada8700f[m
Author: Alexis Cook <alexis.cook@gmail.com>
Date:   Fri Jul 6 13:36:24 2018 -0500

    Initial commit
