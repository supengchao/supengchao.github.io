test
test
I am a slow walker But I never walk backwards
The man who has made up his mind to win will never say "impossible"

2016 年 1 月 28 日，Google 公司 Deepmind 团队在《Nature》杂志上发表重磅学术论文，正式介绍在公平对局条件下以 5:0 成绩击败欧洲围棋冠军樊麾的人工智能程序——AlphaGo。

2016 年 3 月 9 日，AlphaGo 与韩国九段顶尖围棋高手李世石进行“人机大战”，最终以 4:1 战绩一战成名。

2017 年 5 月，在中国乌镇围棋峰会上，AlphaGo 与世界排名第一的围棋冠军柯洁对战，以 3:0 的比分获胜。在横扫了世界围棋界之后，AlphaGo 就此宣布退役。虽 AlphaGo 将不再参加围棋比赛，但其开发团队 DeepMind 公司并没有停下研究的脚步。

三天自学成才的最强阿尔法狗——AlphaGo Zero

就在昨天，10 月 18 日，DeepMind 团队再次带着 AlphaGo 强势归来，于《Nature》杂志上发布一篇名为《Mastering the game of Go without human knowledge》的论文，正式宣布最强版阿尔法狗诞生，命名——AlphaGo Zero。

对此 AlphaGo 项目首席研究员 大卫·席尔瓦表示：“AlphaGo Zero 是世界上最强大的围棋程序，胜过以往所有的 AlphaGo 版本，尤其值得一提的是，它击败了曾经战胜世界围棋冠军李世石的 AlphaGo 版本，成绩为 100:0。”

过去所有版本的 AlphaGo，都是利用人类数据训练开始，它们被告知人类高手具体如何下棋。而最强版 AlphaGo Zero 不使用任何人类数据，独门秘籍是从一张白纸到满腹经纶，花费了三天的时间完全自学成才，它使用了更多原理和算法，这样提高了计算效率。而在 3 天内——也就是 AlphaGo Zero 在击败 AlphaGo Lee 之前，曾进行过 490 万次自我对弈练习。 相比之下，AlphaGo Lee 的训练时间长达数月之久。AlphaGo Zero 不仅发现了人类数千年来已有的许多围棋策略，还设计了人类玩家以前未知的的策略。


AlphaGo-Zero 的训练时间轴

之所以 AlphaGo Zero 比通过人类数据学习获得更好的成绩，是因为：

首先，AlphaGo Zero 仅用棋盘上的黑白子作为输入，而前代则包括了小部分人工设计的特征输入。

其次，AlphaGo Zero 仅用了单一的神经网络。在此前的版本中，AlphaGo用到了“策略网络”来选择下一步棋的走法，以及使用“价值网络”来预测每一步棋后的赢家。而在新的版本中，这两个神经网络合二为一，从而让它能得到更高效的训练和评估。

第三，AlphaGo Zero 并不使用快速、随机的走子方法。在此前的版本中，AlphaGo用的是快速走子方法，来预测哪个玩家会从当前的局面中赢得比赛。相反，新版本依靠地是其高质量的神经网络来评估下棋的局势。


AlphaGo 不同版本所需的 GPU/TPU 资源


AlphaGo 几个版本的排名情况

上述差异均有主于提高系统的性能和通用性，但使最关键的仍是算法上的改进。如今的 AlphaGo Zero 不再受人类知识限制，只用 4 个 TPU。

而此前的 AlphaGo 版本，结合了数百万人类围棋专家的棋谱，以及强化学习的监督学习进行了自我训练。在战胜人类围棋职业高手之前，它经过了好几个月的训练，依靠的是多台机器和 48 个 TPU。

技术实现

新方法使用了一个具有参数θ的深层神经网络fθ。这个神经网络将棋子的位置和历史状态s作为输入，并输出下一步落子位置的概率，用， (p, v) = fθ(s)表示。落子位置概率向量p代表每一步棋（包括不应手）的概率，数值v是一个标量估值，代表棋手下在当前位置s的获胜概率。

AlphaGo Zero 的神经网络通过新的自我对弈数据进行训练，在每个位置s，神经网络fθ都会进行蒙特卡洛树（MCTS）搜索，得出每一步落子的概率π。这一落子概率通常优于原始的落子概率向量p，在自我博弈过程中，程序通过基于蒙特卡洛树的策略来选择下一步，并使用获胜者z作为价值样本，这一过程可被视为一个强有力的评估策略操作。在这一过程中，神经网络参数不断更新，落子概率和价值 (p,v)= fθ(s)也越来越接近改善后的搜索概率和自我对弈胜者 (π， z)，这些新的参数也会被用于下一次的自我对弈迭代以增强搜索的结果，下图即为自我训练的流程图。


AlphaGo Zero 强化学习下的自我对弈流程图

成果

DeepMind 团队在官方博客上称，Zero 用更新后的神经网络和搜索算法重组，随着训练地加深，系统的表现一点一点地在进步。自我博弈的成绩也越来越好，同时，神经网络也变得更准确。


AlphaGo Zero 习得知识的过程

最后大卫·席尔瓦表示：对于希望利用人工智能推动人类社会进步为使命的 DeepMind 来说，围棋并不是 AlphaGo 的终极奥义，他们的目标始终是要利用 AlphaGo 打造通用的、探索宇宙的终极工具。

《Mastering the game of Go without human knowledge》的论文下载地址：https://deepmind.com/documents/119/agz_unformatted_nature.pdf
DeepMind 放出 AlphaGo Zero 的 80 局棋谱，下载地址：https://www.nature.com/nature/journal/v550/n7676/extref/nature24270-s2.zip

看到如此逆天的阿尔法狗，作为紧随技术潮流的小姐姐再也坐不住了，是时候放出终极大招，帮助我们技术同行者共同学习 AlphaGo 的核心技术——机器学习。

首先结合拥有世界上最大的开源数据集的 GitHub 开始我们的第一步。如今 GitHub 的数据科学团队开始探索如何使用机器学习来使开发人员拥有更好的体验。接下来，我们将与数学科学家 Omoju Miller 共同探索机器学习的基础知识以及从开源的项目中学习。

什么是机器学习？
 
机器学习是一门关于算法的研究，使用数据去学习、推广和预测。机器学习令人兴奋之处在于，数据越多，算法越能改进其预测。举个例子，当我家人开始使用语音而不是以往的打字输入进行搜索时，一开始机器需要一段时间来识别我们说的话，但使用语音搜索一个星期以后，算法的语音检测能力已经足够好了，语音输入至今都是我家的主要搜索形式。
 
机器学习从核心上来说不是一个新概念。机器学习是由 IBM 计算机科学家 Arthur Samuel 在 1959 年创造的，自 20 世纪 80 年代以来被广泛应用于软件。
 
随着人们从物理领域转移到数字领域，我们可以从他们留下的数据中学习。
 
举个自己的例子，二十一世纪初我学术研究的一部分是建立神经网络。学习和构建这些算法缺乏真正的商业应用，缺少取得大量数据的途径。随着人们从物理领域转移到数字领域，他们留下的数字足迹可以让我们从中学习。随着全球约有 30 亿人使用互联网，这些足迹积累了惊人的数据量。
 
这些数据存储就是我们所说的“大数据”。 随着大数据的出现，机器学习算法终于能够从学术界转向产业，助力于为消费者提供大量价值的产品。然而收集和获取数据只是构建机器学习数据产品（如搜索引擎和推荐系统）难题的一部分。直到最近，软件程序员、数据科学家和统计师都缺乏利用、清理和打包这些大量数据集的工具，以便其他应用程序的使用。
 
现在通过 Amazon Web Services 和 Hadoop 等工具，我们可以更好更经济有效地管理信息。这些工具为从大数据集中获得价值开辟了新的可能性。
 
Amazon Web Services：https://github.com/aws
apache / Hadoop（Mirror of Apache Hadoop镜像）：https://github.com/apache/hadoop

近年来，机器学习已经扩展到各种新的应用范围。我们尝试用算法去做各种各样的事情，从模式识别到玩游戏甚至到“做梦”。
 
jbhuang0604 / awesome-computer-vision（计算机视觉资源列表）：https://github.com/jbhuang0604/awesome-computer-vision

即使机器学习如今已经有了令人兴奋的发展，目前只是在很多可能性的开始阶段。
 
机器学习如何工作？
 
要想深入了解机器学习，可以将这一过程分为三个部分：输入、算法、输出。
 
输入：驱动机器学习的数据
 
输入是训练和算法需要的数据集。从源代码到统计数据，数据集可以包含任何东西：
 
GSA / data（美国总务管理局数据）：https://github.com/GSA/data
GoogleTrends / data（所有开源数据的索引）：https://github.com/GoogleTrends/data

nationalparkservice / data（美国国家公园管理局非官方数据存储）：https://github.com/nationalparkservice/data

fivethirtyeight / data（FiveThirtyEight上故事与互动背后的数据和代码）：https://github.com/fivethirtyeight/data

beamandrew / medical-data：https://github.com/beamandrew/medical-data

src-d / awesome-machine-learning-on-source-code（机器学习相关有趣的链接和研究论文应用于源代码）：https://github.com/src-d/awesome-machine-learning-on-source-code

我们需要这些输入来训练机器学习算法，因此发现和生成高质量的数据集是当今机器学习面临的最大挑战之一。
 
算法：如何处理和分析数据
 
算法能将数据转化为观点。
 
机器学习算法使用数据来执行特定任务。 最常见的算法类型有：
 
1. 监督学习使用已经标注和结构化的训练数据。通过指定一组输入和所需的输出，机器将学习如何成功识别并将其映射。
 
例如，在决策树学习中，通过将一组决策规则应用于输入数据来预测值：
 
igrigorik / decisiontree（基于ID3的机器学习决策树算法的实现）：https://github.com/igrigorik/decisiontree

2. 无监督学习是使用非结构化数据来发现模式和结构的过程。监督学习可能使用excel表格作为其数据输入，而无监督学习可能用来理解书籍或博客。
 
例如，无监督学习是自然语言处理（NLP）中的流行方法：
 
keon / awesome-nlp（NLP的专用资源列表）：https://github.com/keon/awesome-nlp

3. 强化学习用算法来实现目标。算法朝着目标执行任务，通过奖励和惩罚使之学习正确的方法。
 
例如，强化学习可能用于开发自动驾驶汽车或教机器人如何制造一件物品。
 
openai / gym（一种用于开发和比较强化学习算法的工具包）：https://github.com/openai/gym

aikorea / awesome-rl（强化学习资源）：https://github.com/aikorea/awesome-rl

以下是实践中的几个算法实例：

umutisik / Eigentechno（音乐循环主成分分析）：https://github.com/umutisik/Eigentechno

jpmckinney / tf-idf-similarity（使用tf*idf（词频和逆向文件频率）来计算文本之间相似度）：https://github.com/jpmckinney/tf-idf-similarity

scikit-learn-contrib / lightning（Python中的规模线性分类、回归和排序）：https://github.com/scikit-learn-contrib/lightning

gwding / draw_convnet：https://github.com/gwding/draw_convnet

一些用于执行这些分析的库和工具包括：

scikit-learn / scikit-learn（Python中的机器学习）：https://github.com/scikit-learn/scikit-learn

tensorflow / tensorflow（使用可扩展机器学习的数据流图进行计算）：https://github.com/tensorflow/tensorflow

Theano / Theano（Theano是一个Python库，可让你高效定义、优化、评估涉及多维数组的数学表达式。它可以使用GPU并执行高效的符号）：https://github.com/Theano/Theano

shogun-toolbox / shogun（将军机器学习工具箱（源代码））：https://github.com/shogun-toolbox/shogun

davisking / dlib（用于在C ++中进行真实世界机器学习和数据分析应用的工具包）：https://github.com/davisking/dlib

apache / incubator-predictionio（PredictionIO是开发人员和机器学习工程师的机器学习服务器，基于Apache Spark，HBase和Spray）：https://github.com/apache/incubator-predictionio

什么是深度学习？深度学习是机器学习的一个子集，使用神经网络来查找数据之间关系。深度学习通过监督学习、无监督学习或强化学习来实现其目标。
 
在这个链接，你可以直接在浏览器中体验神经网络：https://github.com/collections/machine-learning
 
虽然深度学习已经存在了数十年，但由于2005年左右图形处理单元（GPU）的创新，神经网络才成为可能。GPU最初是为了在3D游戏环境中渲染像素而开发的，但已经在训练神经网络算法中发现GPU的一个新作用。
 
输出
 
输出是最终结果。输出可能是识别红色符号的模式，可能是判断网页论调正面或负面的情感分析，或者是有置信区间的一个预测分数。
 
在机器学习中，输出可以是任何事物。产生输出的几种方法包括：
 
分类：为数据集中的每一项生成输出值
回归：通过已有数据来预测所考虑变量的最可能值
聚类：将数据分组成相似模式

以下是机器学习的一些实际例子：

deepmind / pysc2（星际争霸II学习环境）：https://github.com/deepmind/pysc2

计算生物学家利用深度学习来理解 DNA：

gokceneraslan / awesome-deepbio（计算生物学领域深度学习应用清单）：https://github.com/gokceneraslan/awesome-deepbio
 
使用Tensorflow进行法语到英语翻译：

buriburisuri / ByteNet（使用DeepMind的ByteNet进行法语到英语机器翻译）：https://github.com/buriburisuri/ByteNet

万事俱备，你准备好开始了吗？

GitHub 上网友整理的机器学习资源供你选择，你也可以将自己的资源添加到这些列表中。
 
机器学习：

josephmisiti / awesome-machine-learning（机器学习框架、库和软件列表）：https://github.com/josephmisiti/awesome-machine-learning

ujjwalkarn / Machine-Learning-Tutorials（机器学习和深度学习教程、文章等资源）：https://github.com/ujjwalkarn/Machine-Learning-Tutorials
 
深度学习：

ChristosChristofidis / awesome-deep-learning（深度学习教程、项目和社区列表）：https://github.com/ChristosChristofidis/awesome-deep-learning
fastai / courses（fast.ai课程）：https://github.com/fastai/courses
 
Tensorflow：

jtoy / awesome-tensorflow（http://tensorflow.org专用资源）：https://github.com/jtoy/awesome-tensorflow

nlintz / TensorFlow-Tutorials（使用Google TensorFlow框架的简单教程）：https://github.com/nlintz/TensorFlow-Tutorials

pkmital / tensorflow_tutorials（从基础到更有趣的Tensorflow应用）：https://github.com/pkmital/tensorflow_tutorials
