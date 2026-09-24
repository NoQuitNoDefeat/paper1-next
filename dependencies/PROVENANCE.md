# 依赖与复用代码来源

| 内容 | 来源 | 版本 |
| --- | --- | --- |
| ns-3 | https://gitlab.com/nsnam/ns-3-dev.git | ns-3.48，commit d2add90b452d600cfb4859baed8e9ea633519447（见 ns3.lock） |
| ns3-ai | https://github.com/hust-diangroup/ns3-ai.git | commit b8c9858294b1d6a7f122b5154a3ce25057a54740，加 3 个补丁（见 ns3-ai.lock） |
| `patches/`、两个 lock 文件 | paper1-baseline/dependencies | paper1-baseline commit 84510214810619bfda0b5ddb072a54eddaa4decd |
| `simulator/ns-3-external-contrib/fanet-scheduler` | paper1-baseline/simulator/ns-3-external-contrib/fanet-scheduler | 同上；复制于 2026-09-24，之后在本项目内修改，改动见 git 历史 |

paper1-baseline 只读引用，未做任何修改。ns-3 与 ns3-ai 的源码树由 `tools/ns3/setup.sh` 在 `simulator/ns-3-dev` 生成（不进版本库）。
