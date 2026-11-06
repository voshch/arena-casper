# arena-casper

CASPER++, the multi-agent task allocation architecture by Leticia Berto (University of
Manchester, [leticiaberto/CASPER_MAS](https://github.com/leticiaberto/CASPER_MAS)), as two
packages: its core and an Arena task layer. Its own distributed runtime on ROS 2 Humble and
Gazebo stays in CASPER_MAS.

| Package | What it is |
| --- | --- |
| [casper](casper/README.md) | The core: goal graphs, team skills and preferences, the four allocation modes, workload rebalancing and the task schedule, plain Python 3.10 and later |
| [arena_casper](arena_casper/README.md) | The `casper` task layer of Arena: the team file keyed by robot names and the episode loop dispatching each robot's tasks |

In Arena, `arena feature tasks add casper` installs it and `task_layer:=casper` selects it.
colcon builds both packages.
