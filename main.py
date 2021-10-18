from redbaron import RedBaron, TupleNode, ExceptNode, IfelseblockNode, IfNode
from redbaron import CallNode, DefNode
from redbaron import ForNode
from redbaron import NameNode
from redbaron import AtomtrailersNode
from redbaron import WhileNode
import argparse
import re
import os
from redbaron import TryNode
import subprocess

file_list = []


def walkFile(file):
    for root, dirs, files in os.walk(file):
        # 遍历文件
        for f in files:
            if f.endswith('.py'):
                file_list.append(os.path.join(root, f))


def update_pos(p, loc):
    i = 0
    while i < len(pos):
        if p <= pos[i]:
            pos[i] += loc
        i += 1

def get_variable_to_report(str): #获取变量名主体
    red = RedBaron(str)
    node = red[0]
    if isinstance(node, NameNode):
        return node.dumps()
    elif isinstance(node, AtomtrailersNode):
        getitemnode = node.find('GetitemNode')
        if getitemnode is not None:
            parent_getitemnode = getitemnode.parent
            index_getitemnode = parent_getitemnode.index(getitemnode)
            if isinstance(parent_getitemnode.value[index_getitemnode - 1], NameNode):
                return parent_getitemnode.value[index_getitemnode - 1].dumps()
        else:
            callnode = node.find('CallNode')
            arguments = []
            if callnode is not None:
                callarguments = callnode.find_all('CallArgumentNode')
                if callarguments is not None:
                    for callargument in callarguments:
                        if (isinstance(callargument.value, NameNode)):
                            arguments.append(callargument.value.dumps())
                    return arguments

def add_flag_arg():  # 用户从命令行输入控制的Flag
    add_argument_node = red.find("name", value='add_argument')
    parent_of_add_argument_node = add_argument_node.parent
    index_of_add_argument_node = parent_of_add_argument_node.index(add_argument_node)
    #     print(type(parent_of_fitnode.value[index_of_fitnode+1]))
    while not isinstance(parent_of_add_argument_node.value[index_of_add_argument_node + 1], CallNode):
        add_argument_node = red.find("name", value='add_argument')
    add_argument_node = add_argument_node.parent
    parent_of_add_argument_node = add_argument_node.parent
    index_of_add_argument_node = parent_of_add_argument_node.index(add_argument_node)
    parent_of_add_argument_node.value.insert(index_of_add_argument_node,
                                             "parser.add_argument('--nni', action='store_true', default=False, help='whether to tune the pruners using NNi tuners')")


def add_import():
    first_import_node = red.find('ImportNode')
    index = red.index(first_import_node)
    red.insert(index,"NNI_FLAG = False")
    import_node = RedBaron("if NNI_FLAG:\n    import automl")
    red.insert(index+1,import_node.dumps())
    red.insert(index, "#--------------------------------------")
    p = (red[index].absolute_bounding_box.top_left.line)
    update_pos(p, 5)
    pos.append(p)
    lines_of_codes.append(5)
    contents_inserted.append(import_node.dumps())
    # print(p)


def insert_path_for_pbt(path):
    model_init_nodes = red.find_all("assignment", target=lambda x: x.dumps() == "model")
    if model_init_nodes:
        all_in_same_if = True
        ifnode = None
        if len(model_init_nodes) > 1:
            for node in model_init_nodes:  # 判断是否在同一个if中
                if ifnode is None:
                    tmp = node.parent
                    while not isinstance(tmp, IfelseblockNode):
                        tmp = tmp.parent
                        if tmp == red:
                            break
                    if tmp is red:
                        all_in_same_if = False
                        break
                    else:
                        ifnode = tmp
                else:
                    tmp = node.parent
                    while tmp is not ifnode:
                        tmp = tmp.parent
                        if tmp == red:
                            break
                    if tmp is not ifnode:
                        all_in_same_if = False
                        break
        if all_in_same_if and ifnode is not None:
            parent = ifnode.parent
            index = parent.index(ifnode)
            parent.value.insert(index + 1,
                                                 "if os.path.isfile(load_checkpoint_path):\n    model_state_dict = torch.load(load_checkpoint_path)\n    model.load_state_dict(model_state_dict)")
            parent.value.insert(index + 1,
                                                 "load_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth')")
            parent.value.insert(index + 1, "#--------------------------------------")
            p = (parent.value[index + 1].absolute_bounding_box.top_left.line)
            update_pos(p, 6)
            pos.append(p)
            lines_of_codes.append(6)
            contents_inserted.append(
                "if os.path.isfile(load_checkpoint_path):\n    model_state_dict = torch.load(load_checkpoint_path)\n    model.load_state_dict(model_state_dict)\nload_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth')")

        else:
            final_model_init = model_init_nodes[-1]
            parent = final_model_init.parent
            while not isinstance(parent, IfNode):
                if parent == red:
                    break
                parent = parent.parent
            if isinstance(parent, IfNode) and isinstance(parent.parent, IfelseblockNode):
                final_model_init = parent.parent
            elif isinstance(parent, IfNode):
                final_model_init = parent
            # print(final_model_init.dumps())
            # print(final_model_init.parent.dumps())
            index_final_model_init = final_model_init.parent.value.index(final_model_init)

            final_model_init.parent.value.insert(index_final_model_init + 1,
                                                 "if os.path.isfile(load_checkpoint_path):\n    model_state_dict = torch.load(load_checkpoint_path)\n    model.load_state_dict(model_state_dict)")
            final_model_init.parent.value.insert(index_final_model_init + 1,
                                                 "load_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth')")
            final_model_init.parent.value.insert(index_final_model_init + 1, "#--------------------------------------")

            # print(final_model_init.parent.value[index_final_model_init + 1])
            p = (final_model_init.parent.value[index_final_model_init + 1].absolute_bounding_box.top_left.line)
            update_pos(p, 6)
            pos.append(p)
            lines_of_codes.append(6)
            contents_inserted.append("if os.path.isfile(load_checkpoint_path):\n    model_state_dict = torch.load(load_checkpoint_path)\n    model.load_state_dict(model_state_dict)\nload_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth')")
        #插入save nodes
        final_model_init = model_init_nodes[-1]
        parent = final_model_init.parent
        while not isinstance(parent, DefNode):
            if parent == red:
                break
            parent = parent.parent
        if parent == red:
            parent.insert(len(parent),
                                   "#--------------------------------------")
            parent.insert(len(parent),
                                   "save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')")
            parent.insert(len(parent),
                                   "if not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
            p = (parent[len(parent) - 3].absolute_bounding_box.top_left.line)

            update_pos(p, 6)
            pos.append(p)
            lines_of_codes.append(6)
            contents_inserted.append(
                "save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')\nif not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
        else:
            parent.value.insert(len(parent.value),
                          "#--------------------------------------")
            parent.value.insert(len(parent.value),
                          "save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')")
            parent.value.insert(len(parent.value),
                          "if not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
            p = (parent.value[len(parent.value) - 3].absolute_bounding_box.top_left.line)

            update_pos(p, 6)
            pos.append(p)
            lines_of_codes.append(6)
            contents_inserted.append(
                "save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')\nif not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
    else: #不存在model的assignment
        train_node = red.find("name", value='train')
        if train_node is not None:
            parent_train = train_node.parent
            index_of_train_node = parent_train.index(train_node)
            while not isinstance(parent_train.value[index_of_train_node + 1], CallNode):
                train_node = red.find("name", value='train')
            train_node = train_node.parent
            index_train = train_node.parent.index(train_node)
            train_node.parent.value.insert(index_train,
                                           "if NNI_FLAG:\n    load_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth')\n    if os.path.isfile(load_checkpoint_path):\n        model_state_dict = torch.load(load_checkpoint_path)\n        model.load_state_dict(model_state_dict)")
            train_node.parent.value.insert(index_train,"#--------------------------------------")
            p = (train_node.parent.value[index_train].absolute_bounding_box.top_left.line)
            update_pos(p, 7)
            pos.append(p)
            lines_of_codes.append(7)
            contents_inserted.append("if NNI_FLAG:\n    load_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth')\n    if os.path.isfile(load_checkpoint_path):\n        model_state_dict = torch.load(load_checkpoint_path)\n        model.load_state_dict(model_state_dict)")

    # main_func = red.find("DefNode", name='main')
    # if main_func:
    #     main_func.value.insert(len(main_func.value),
    #                            "#--------------------------------------")
    #     main_func.value.insert(len(main_func.value),
    #                            "save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')")
    #     main_func.value.insert(len(main_func.value),
    #                            "if not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
    #     p = (main_func.value[len(main_func.value)-3].absolute_bounding_box.top_left.line)
    #
    #     update_pos(p, 6)
    #     pos.append(p)
    #     lines_of_codes.append(6)
    #     contents_inserted.append("save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')\nif not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
    # else:
    #     main_func = red.find("DefNode", name='run_training_loop')
    #     if main_func:
    #         main_func.value.insert(len(main_func.value),
    #                                "#--------------------------------------")
    #         main_func.value.insert(len(main_func.value),
    #                               "if NNI_FLAG:\n    save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')\n    if not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")
    #         p = (main_func.value[len(main_func.value)-2].absolute_bounding_box.top_left.line)
    #         # print(p)
    #         update_pos(p, 7)
    #         pos.append(p)
    #         lines_of_codes.append(7)
    #         contents_inserted.append("if NNI_FLAG:\n    save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth')\n    if not os.path.exists(params['save_checkpoint_dir']):\n    os.makedirs(params['save_checkpoint_dir'])\n    torch.save(model.state_dict(), save_checkpoint_path)")

    # main_node = red.find("IfNode", test=lambda x: x.dumps() == "__name__ == '__main__'")
    # if not isinstance(main_node.value[0], TryNode):
    #     main_node.value.insert(0, "if NNI_FLAG:\n    automl.set_pbt_path('%s')" % path)
    #     print(main_node.value[0].absolute_bounding_box)
    # else:
    #     main_node.value[0].value.insert(0, "if NNI_FLAG:\n    automl.set_pbt_path('%s')" % path)
    #     print(main_node.value[0].value[0].absolute_bounding_box)


def insert_after_assignment(names=[], stat=""):
    for name in names:
        assignnodes = red.find_all("assignment", target=lambda x: x.dumps() == name)
        for node in assignnodes:
            if (node.parent == red):
                index = (red.index(node))
            else:
                index = (node.parent.value.index(node))
            node.parent.insert(index + 1,  "#--------------------------------------")
            node.parent.insert(index + 2, stat)
            p = (node.parent.value[index + 2].absolute_bounding_box.top_left.line)
            update_pos(p, 3)
            pos.append(p)
            lines_of_codes.append(3)
            contents_inserted.append(stat)

        tuples = red.find_all("tuple")
        for tuple in tuples:
            paras = []
            for value in tuple.value:
                paras.append(value.dumps())
            if (name in paras):
                assigntuples = red.find_all("assignment", target=lambda tar: tar == tuple)
                #             print(assigntuples)
                for node in assigntuples:
                    if (node.parent == red):
                        index = (red.index(node))
                    else:
                        index = (node.parent.value.index(node))
                    node.parent.insert(index + 1, "#--------------------------------------")
                    node.parent.insert(index + 2, stat)
                    p = (node.parent.value[index + 2].absolute_bounding_box.top_left.line)
                    update_pos(p, 3)
                    pos.append(p)
                    lines_of_codes.append(3)
                    contents_inserted.append(stat)


def insert_nni_final_with_fit(names=[]):
    for name in names:
        assignnodes = red.find_all("assignment", target=lambda x: x.dumps() == get_variable_to_report(name))
        assignnodes_tuples = red.find_all('Assignment', target=lambda x: (isinstance(x, TupleNode) and x.find('NameNode', value=get_variable_to_report(name)) is not None))
        if assignnodes and assignnodes_tuples:
            assignnodes.extend(assignnodes_tuples)
        elif assignnodes:
            assignnodes = assignnodes
        elif assignnodes_tuples:
            assignnodes = assignnodes_tuples

        if assignnodes:
            cur = len(assignnodes) - 1
            while cur >= 0:
                assignnode = assignnodes[cur]
                cur -= 1
                parent = assignnode.parent
                fit = parent.find('Name', value = 'fit')
                if fit:
                    fit_line = (fit.absolute_bounding_box.top_left.line)
                    assignnode_line = assignnode.absolute_bounding_box.top_left.line
                    if(fit_line < assignnode_line):
                        stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name, name)
                        if (assignnode.parent == red):
                            index = (red.index(assignnode))
                        else:
                            index = (assignnode.parent.value.index(assignnode))
                        assignnode.parent.insert(index + 1, "#--------------------------------------")
                        assignnode.parent.insert(index + 2, stat)
                        p = (assignnode.parent.value[index + 2].absolute_bounding_box.top_left.line)
                        update_pos(p, 4)
                        pos.append(p)
                        lines_of_codes.append(4)
                        contents_inserted.append(stat)
                        break




def get_callback_classes():
    fitnodes = red.find_all("name", value='fit')
    for fitnode in fitnodes:
        parent_of_fitnode = fitnode.parent
        index_of_fitnode = parent_of_fitnode.index(fitnode)
        fit_next_node = parent_of_fitnode.value[index_of_fitnode + 1]
        callback_classes = []
        if isinstance(fit_next_node, CallNode):
            #             model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
            #         validation_data=(x_test, y_test), callbacks=[SendMetrics()])
            callbacks_node = fit_next_node.find("call_argument",
                                                target=lambda x: x is not None and x.dumps() == "callbacks")
            for arg in callbacks_node.value:
                print(arg.value[0])
                callback_classes.append(arg.value[0].value)
            return callback_classes
    return []


def insert_nni_intermediate_with_fit(name):
    def_epoch_end = red.find("DefNode", name='on_epoch_end')
    # callbacks_classes = get_callback_classes()
    # if (def_epoch_end.parent.name in callbacks_classes):
    def_epoch_end.insert(1, "#----------------------------")
    def_epoch_end.insert(2, "if NNI_FLAG:\n    automl.report_intermediate_result(%s)" % name)
    p = (def_epoch_end.value[1].absolute_bounding_box.top_left.line)
    update_pos(p, 4)
    pos.append(p)
    contents_inserted.append( "if NNI_FLAG:\n    automl.report_intermediate_result(%s)" % name)
    lines_of_codes.append(4)


def insert_report_with_batch_loop(names = [], final = ''):
    for name in names:
        node_in_try = []
        stat_for_node_in_try = []
        batch_loop_node = None
        parent_of_parent = None
        assign_after_loop = False  # 循环后是否重新赋值
        assignnodes = red.find_all("assignment", target=lambda x: x.dumps() == get_variable_to_report(name))
        if not assignnodes:
            assignnodes = red.find_all('Assignment', target=lambda x: (isinstance(x, TupleNode) and x.find('NameNode', value = get_variable_to_report(name)) is not None))
        for node in assignnodes:
            is_in_try = False
            parent = node.parent
            while not isinstance(parent, TryNode):#判断是否在Try结构中
                parent = parent.parent
                if parent == red:
                    break
            if parent is not red:
                is_in_try = True
                node_in_try.append(node)

            parent = node.parent
            while not isinstance(parent, ForNode) and not isinstance(parent, WhileNode):
                parent = parent.parent
                if parent == red:
                    break
            if parent is not red:
                parent_of_parent = parent.parent
                while not isinstance(parent_of_parent, ForNode):
                    parent_of_parent = parent_of_parent.parent
                    if parent_of_parent == red:
                        break
            if batch_loop_node is None and parent is red:  # 不在循环内且第一次赋值为初始化：
                continue
            if parent_of_parent is not None and parent_of_parent is not red:  # 两层循环
                tem = parent.parent
                index = len(tem.value)
                tem.value.insert(index, "\n")
                tem.value.insert(index + 1, "#-----------------------")
                tem.value.insert(index + 2,
                                 "if NNI_FLAG:\n    automl.report_intermediate_result('%s', %s)" % (name ,name))
                p = (tem.value[index].absolute_bounding_box.top_left.line)
                update_pos(p, 5)
                pos.append(p)
                contents_inserted.append( "if NNI_FLAG:\n    automl.report_intermediate_result('%s', %s)" % (name ,name))
                lines_of_codes.append(5)
                #                     print(tem.dumps())
                tem_1 = parent_of_parent.parent
                index = len(tem_1.value)
                tem_1.value.insert(index, "\n")
                tem_1.value.insert(index + 1, "#-----------------------")
                tem_1.value.insert(index + 2, "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name ,name))
                p = (tem_1.value[index].absolute_bounding_box.top_left.line)
                update_pos(p, 5)
                pos.append(p)
                lines_of_codes.append(5)
                contents_inserted.append( "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name ,name))
                break

            elif batch_loop_node is None and parent is not red: #循环内第一次赋值
                batch_loop_node = parent
                stat = "if NNI_FLAG:\n    automl.report_intermediate_result('%s', %s)" % (name ,name)
                if is_in_try:
                    stat_for_node_in_try.append(stat)
                    continue
            elif batch_loop_node is not None and node.parent == batch_loop_node.parent and (
                    node.parent.index(node) > node.parent.index(batch_loop_node)):  # 循环后重新赋值
                assign_after_loop = True
                stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name ,name)
            elif parent == batch_loop_node:  # 同在循环之内
                assign_after_loop = True
                stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name ,name)
            if (node.parent == red):
                index = (red.index(node))
            else:
                index = (node.parent.value.index(node))

            node.parent.value.insert(index + 1, "\n")
            node.parent.value.insert(index + 2, "#---------------------")
            node.parent.value.insert(index + 3, stat)
            p = (node.parent.value[index + 2].absolute_bounding_box.top_left.line)
            update_pos(p, 5)
            pos.append(p)
            lines_of_codes.append(5)
            contents_inserted.append(stat)
        # print(node_in_try, stat_for_node_in_try)
        for (node, stat) in zip(node_in_try, stat_for_node_in_try): #先插入except，再插入try
            if (node.parent == red):
                index = (red.index(node))
            else:
                index = (node.parent.value.index(node))

            node.parent.value.insert(index + 1, "\n")
            node.parent.value.insert(index + 2, "#---------------------")
            node.parent.value.insert(index + 3, stat)
            p = (node.parent.value[index + 2].absolute_bounding_box.top_left.line)
            update_pos(p, 5)
            pos.append(p)
            lines_of_codes.append(5)
            contents_inserted.append(stat)

        if not assign_after_loop and batch_loop_node is not None:  # 循环后未赋值
            index = batch_loop_node.parent.index(batch_loop_node)
            if final is '':
                final = name
            stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (final ,final)
            batch_loop_node.parent.insert(index + 1, "#-----------------------")
            batch_loop_node.parent.insert(index + 2, stat)
            p = (batch_loop_node.parent.value[index + 1].absolute_bounding_box.top_left.line)
            update_pos(p, 4)
            pos.append(p)
            lines_of_codes.append(4)
            contents_inserted.append(stat)
        cur_DefNode = []
        for node in assignnodes:
            parent = node.parent
            while not isinstance(parent, DefNode):
                if parent == red:
                    break
                parent = parent.parent
            if isinstance(parent, DefNode) and parent not in cur_DefNode:
                cur_DefNode.append(parent)
                return_node = parent.find_all('return')
                for rtnode in return_node:
                    parent_rtnode = rtnode.parent
                    index = parent_rtnode.value.index(rtnode)
                    parent_rtnode.value.insert(index, "\n")
                    parent_rtnode.value.insert(index + 1, "#-----------------------")
                    parent_rtnode.value.insert(index + 2,
                                       "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name, name))
                    p = (parent_rtnode.value[index].absolute_bounding_box.top_left.line)
                    update_pos(p, 5)
                    pos.append(p)
                    lines_of_codes.append(5)
                    contents_inserted.append("if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name, name))



def update_params(params):
    inserted = False
    main_node = red.find("IfNode", test=lambda x: x.dumps() == "__name__ == \"__main__\"")
    #     update = RedBaron("if isinstance(%s, dict) and %s.nni:\n    %s = %s.update(tuner_params)\nelif isinstance(%s, argparse.Namespace) and %s.nni:\n    %s = vars(%s)\n    %s = %s.update(tuner_params)\n    %s = argparse.Namespace(**%s)" % (params, params, params, params, params, params, params, params, params, params, params, params))
    update = RedBaron("if NNI_FLAG:\n    automl.update_parameter(%s)" % params)
    if main_node is not None:  # 存在程序入口__main__
        init_param_nodes = main_node.find_all("assignment", target=lambda x: get_variable_to_report(x.dumps()) == params)
        print(init_param_nodes)
        if init_param_nodes:
            init_param_node = init_param_nodes[-1]
            index_param = init_param_node.parent.value.index(init_param_node)
            init_param_node.parent.value.insert(index_param + 1, "#------------------------")
            init_param_node.parent.value.insert(index_param + 2, update.dumps())  # 首先插入更新
            p = (init_param_node.parent.value[index_param + 1].absolute_bounding_box.top_left.line)
            update_pos(p, 4)
            pos.append(p)
            lines_of_codes.append(4)
            contents_inserted.append(update.dumps())
            inserted = True
            main_value = main_node.value
        else:
            init_param_nodes = red.find_all("assignment", target=lambda x: get_variable_to_report(x.dumps()) == params)
            if init_param_nodes:
                init_param_node = init_param_nodes[-1]
                index_param = init_param_node.parent.value.index(init_param_node)
                init_param_node.parent.value.insert(index_param + 1, "#------------------------")
                init_param_node.parent.value.insert(index_param + 2, update.dumps())
                p = (init_param_node.parent.value[index_param + 1].absolute_bounding_box.top_left.line)
                update_pos(p, 4)
                pos.append(p)
                contents_inserted.append(update.dumps())
                lines_of_codes.append(4)
                inserted = True
            else:
                pass
    if inserted == False:
        init_param_nodes = red.find_all("assignment", target=lambda x: get_variable_to_report(x.dumps()) == params)
        # print(init_param_nodes)
        if init_param_nodes:
            init_param_node = init_param_nodes[-1]
            if init_param_node.parent is not red:
                index_param = init_param_node.parent.value.index(init_param_node)
            else:
                index_param = init_param_node.parent.index(init_param_node)

            if init_param_node.parent is not red:
                init_param_node.parent.value.insert(index_param + 1, "#------------------------")
                init_param_node.parent.value.insert(index_param + 2, update.dumps())
                p = (init_param_node.parent.value[index_param + 1].absolute_bounding_box.top_left.line)
            else:
                init_param_node.parent.insert(index_param + 1, "#------------------------")
                init_param_node.parent.insert(index_param + 2, update.dumps())
                p = (init_param_node.parent[index_param + 1].absolute_bounding_box.top_left.line)
            update_pos(p, 4)
            pos.append(p)
            contents_inserted.append(update.dumps())
            lines_of_codes.append(4)
        else:
            def_arg = red.find('def_argument',target = lambda x: (x.dumps()) == params)
            print(params)
            if def_arg:
                def_arg_parent = def_arg.parent
                if isinstance(def_arg.parent, DefNode):
                    def_arg_parent.value.insert(0, "#------------------------")
                    def_arg_parent.value.insert(1, update.dumps())
                    p = (def_arg_parent.value[1].absolute_bounding_box.top_left.line)
                    update_pos(p, 4)
                    pos.append(p)
                    contents_inserted.append(update.dumps())
                    lines_of_codes.append(4)
            else:
                print("update_params ERROR!")




def insert_only_final_report(names=[]):
    for name in names:
        stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % (name, name)
        assignnodes = red.find_all("assignment", target=lambda x: x.dumps() == get_variable_to_report(name))
        if not assignnodes:
            assignnodes = red.find_all('Assignment', target=lambda x: (isinstance(x, TupleNode) and x.find('NameNode', value = get_variable_to_report(name)) is not None))
        if assignnodes:
            all_in_same_if = True
            ifnode = None
            if len(assignnodes) >1 :
                for node in assignnodes:#判断是否在同一个if中
                    if ifnode is None:
                        tmp = node.parent
                        while not isinstance(tmp, IfelseblockNode):
                            tmp = tmp.parent
                            if tmp == red:
                                break
                        if tmp is red:
                            all_in_same_if = False
                            break
                        else:
                            ifnode = tmp
                    else:
                        tmp = node.parent
                        while tmp is not ifnode:
                            tmp = tmp.parent
                            if tmp == red:
                                break
                        if tmp is not ifnode:
                            all_in_same_if = False
                            break
            if all_in_same_if and ifnode is not None:
                parent = ifnode.parent
                while not (isinstance(parent, ForNode) or isinstance(parent, WhileNode)):
                    parent = parent.parent
                    if parent == red:
                        break
                if parent == red:
                    final_variable_in_loop = False
                else:
                    final_variable_in_loop = True
                    loop_node = parent
                if final_variable_in_loop:
                    # print("adsasdasdasfdasgfasdgsadgdasg")
                    loop_parent = loop_node.parent
                    index = loop_parent.value.index(loop_node)
                    loop_parent.insert(index + 1, "#-----------------------")
                    loop_parent.insert(index + 2, stat)
                    p = (loop_parent[index + 1].absolute_bounding_box.top_left.line)
                    update_pos(p, 4)
                    pos.append(p)
                    contents_inserted.append(stat)
                    lines_of_codes.append(4)
                    break
                else:
                    index = ifnode.parent.index(ifnode)
                    ifnode.parent.insert(index + 1,"#-----------------------")
                    ifnode.parent.insert(index + 2, stat)
                    p = ifnode.parent[index + 1].absolute_bounding_box.top_left.line
                    update_pos(p, 4)
                    pos.append(p)
                    contents_inserted.append(stat)
                    lines_of_codes.append(4)
                    break

            cur = len(assignnodes)-1
            while cur >=0 :
                assignnode = assignnodes[cur]
                cur -= 1
                proper_to_insert = False
                parent = assignnode.parent
                loop_node = None
                while not (isinstance(parent, ForNode) or isinstance(parent, WhileNode)):
                    parent = parent.parent
                    if parent == red:
                        break
                if parent == red:
                    final_variable_in_loop = False
                else:
                    final_variable_in_loop = True
                    loop_node = parent
                if final_variable_in_loop: #寻找最外层循环
                    parent = loop_node.parent
                    while not isinstance(parent, DefNode) and parent is not red:
                        if isinstance(parent, ForNode) or isinstance(parent, WhileNode):
                            loop_node = parent
                        parent = parent.parent
                    proper_to_insert = True

                parent = assignnode.parent
                fornode = parent.find("ForNode")
                if fornode is not None and not proper_to_insert:
                    epochnode = fornode.find("Name", value = 'epoch')
                    batchnode = fornode.find("Name", value = 'batch')
                    if (fornode == parent):
                        proper_to_insert =True
                    elif (fornode.parent == parent ):
                        index_assign = parent.index(assignnode)
                        index_for = parent.index(fornode)
                        if(index_for < index_assign):
                            proper_to_insert = True
                        # and (epochnode is not None or batchnode is not None):

                if proper_to_insert:
                    if final_variable_in_loop:
                        # print("adsasdasdasfdasgfasdgsadgdasg")
                        loop_parent = loop_node.parent
                        index = loop_parent.value.index(loop_node)
                        loop_parent.insert(index + 1, "#-----------------------")
                        loop_parent.insert(index + 2, stat)
                        p = (loop_parent[index + 1].absolute_bounding_box.top_left.line)
                    elif not isinstance(assignnode.parent, IfNode):
                        if (parent == red):
                            index = (red.index(assignnode))
                        else:
                            index = (assignnode.parent.value.index(assignnode))
                        assignnode.parent.insert(index + 1, "#-----------------------")
                        assignnode.parent.insert(index + 2, stat)
                        if (parent == red):
                            p = (assignnode.parent[index + 1].absolute_bounding_box.top_left.line)
                        else:
                            p = (assignnode.parent.value[index + 1].absolute_bounding_box.top_left.line)
                    else:
                        index = assignnode.parent.parent.parent.index(assignnode.parent.parent)
                        assignnode.parent.parent.parent.insert(index + 1, "#-----------------------")
                        assignnode.parent.parent.parent.insert(index + 2, stat)
                        p = (assignnode.parent.parent.parent[index + 1].absolute_bounding_box.top_left.line)
                    update_pos(p, 4)
                    pos.append(p)
                    contents_inserted.append(stat)
                    lines_of_codes.append(4)
                    break


parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')
parser.add_argument('--Model', type=str)
parser.add_argument('--intermediate_necessary', type=str)
parser.add_argument('--arg_to_report', type=str)
parser.add_argument('--params_to_update', type=str)
parser.add_argument('--pbt_path', default='path', type=str)
parser.add_argument('--final_name', default='', type=str)
args = parser.parse_args()
print(args)



def annotate(file):
    pattern = re.compile(r' *.*? *= *\{\*\*.*?\}')
    with open(file, mode='r', encoding = 'utf-8') as f:
        lines = f.readlines()
        line_to_annotate = ''
        line_no = 0
        for line in lines:
            line_no += 1
            match = pattern.match(line)
            if match:
                line_to_annotate = line
                break
        if lines and match:
            src = ''.join(line for line in lines if line is not line_to_annotate)
        else:
            src = ''.join(line for line in lines)
        return  src, line_to_annotate, line_no

def format_except(str):
    lines = str.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith('except'):
            index_try = index-1
            while not lines[index_try].strip().startswith('try:'):
                index_try -= 1
            lines[index] = lines[index_try].replace('try:', lines[index].strip())
        index += 1
    new_str = '\n'.join(line for line in lines)
    return new_str

arg_to_report = get_variable_to_report(args.arg_to_report)

basedir = "F:\\automl_projects\\测试结果9.30\\"
pro_dir = "qe_pipline_5stages_backup"
folder = basedir + pro_dir
walkFile(folder)

main_file = 'qe_pipeline.py'
for file in file_list:
    if file.endswith(main_file):
        entry_file = file[len(basedir):]


cmd = "pycg --package %s %s -o cg.json" % (pro_dir, entry_file)
# cmd = "pycg --package qe_pipline_5stages_backup qe_pipline_5stages_backup/qe_pipeline.py -o cg.json"
subprocess.call(cmd, shell=True, cwd="F:\\automl_projects\\测试结果9.30")

# print(file_list)
# file_list = ['E:\Pyhton workspace\Python AST\src.py']
i = 0
with open(basedir + "cg.json", 'r') as callgraph:
    cg = callgraph.read()

for file in file_list:
    print(file)
    f_name = file.rsplit("\\",1)[1]
    f_name = f_name.strip(".py")
    if cg.find(f_name) != -1:
        is_called = True
    else:
        is_called = False

    # if not (file.endswith(main_file)):
    #     continue
    if not is_called:
        continue

    if file.endswith('_output.py'):
        continue
    pos = []
    lines_of_codes = []
    contents_inserted = []
    i += 1
    print(i, "scanning")
    src, line_to_annotate, line_no = annotate(file)

    # red = RedBaron(src)

    try:
        red = RedBaron(src)#insert try导致结构改变
        # tests = red.find_all('Assignment', target=lambda x: (isinstance(x, TupleNode) and x.find('NameNode', value="prs") is not None))
        # test = tests[-2]
        # stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % ("p", "p")
        # test.parent.value.insert(1, "#-----------------------")
        # test.parent.value.insert(2, stat)
        # test = tests[-1]
        # stat = "if NNI_FLAG:\n    automl.report_final_result('%s', %s)" % ("p", "p")
        # test.parent.value.insert(1, "#-----------------------")
        # test.parent.value.insert(2, stat)
        # print(red.dumps())
    except BaseException:
        continue


    test_report = red.find_all("assignment", target = lambda x: x.dumps() == arg_to_report)
    test_report_tuple = red.find_all('Assignment', target=lambda x: (isinstance(x, TupleNode) and x.find('NameNode', value = arg_to_report) is not None))
    test_report = test_report or test_report_tuple
    test_update = red.find_all("assignment", target=lambda x: x.dumps() == args.params_to_update)
    test_update_tuple = red.find_all('Assignment', target=lambda x: (isinstance(x, TupleNode) and x.find('NameNode', value = args.params_to_update) is not None))
    test_update_def_arg = red.find("def_argument", target=lambda x: x.dumps() == args.params_to_update)
    test_update = test_update or test_update_tuple or test_update_def_arg
    print(bool(test_update))
    print(bool(test_report))
    if not test_report and not test_update:
        continue
    elif not test_update:
        add_import()
        if (args.Model == "pbt"):
            insert_path_for_pbt(args.pbt_path)

        names = []
        names.append(args.arg_to_report)
        final_name = args.final_name

        if (args.Model == "keras"):
            insert_nni_final_with_fit(names)
            if (args.intermediate_necessary == "true"):
                insert_nni_intermediate_with_fit(arg_to_report)
        elif (args.intermediate_necessary == "true"):
            pass
            insert_report_with_batch_loop(names, final_name)
        else:
            insert_only_final_report(names)
    elif not test_report:
        if(file.endswith(main_file)):
            add_import()
            params = args.params_to_update
            print(params)
            update_params(params)
        else:
            continue
    else:
        add_import()
        if(args.Model == "pbt"):
            insert_path_for_pbt(args.pbt_path)

        names = []
        names.append(args.arg_to_report)
        # names = ['acc']
        # insert_nni_final_with_fit(names)
        # insert_nni_intermediate_with_fit()
        final_name = args.final_name

        if(args.Model == "keras"):
            insert_nni_final_with_fit(names)
            if (args.intermediate_necessary == "true"):
                insert_nni_intermediate_with_fit(arg_to_report)
        elif(args.intermediate_necessary == "true"):
            pass
            insert_report_with_batch_loop(names, final_name)
        else:
            insert_only_final_report(names)
        #
        if (file.endswith(main_file)):
            params = args.params_to_update

            update_params(params)
    #
    (filename, extension) = os.path.splitext(file)
    new_file = filename + "_output.py"
    (folderpath, newfile) = os.path.split(new_file)
    result_folder = folder + "\\result_new\\"
    newfile = result_folder + newfile
    logs = result_folder + "logs.txt"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if len(pos) > 1:
        with open(logs, 'a') as logs_writer:
            logs_writer.write(file+'\n'+'---------------------'+'\n')
            insert_cnt = 0
            while insert_cnt < len(pos):
                print(pos[insert_cnt], ' ', lines_of_codes[insert_cnt], ':')
                logs_writer.write(str(pos[insert_cnt])+ ' '+ str(lines_of_codes[insert_cnt])+ ':\n')
                print(contents_inserted[insert_cnt])
                logs_writer.write(contents_inserted[insert_cnt])
                logs_writer.write('\n')
                insert_cnt += 1
            logs_writer.write('\n')
            print(pos)
            print(lines_of_codes)
        index = 0
        offset = 0
        while index < len(pos):
            if(pos[index] <= line_no):
                offset += lines_of_codes[index]
            index += 1
        line_no += offset
        # print(red.dumps())

        with open(newfile, 'w') as writer:
            if line_to_annotate is not '':
                new_lines = red.dumps().splitlines(True)
                new_lines.insert(line_no-1, line_to_annotate)
                dst = ''.join(new_line for new_line in new_lines)
                # print(dst)
                writer.write(dst)
            else:
                # print(red.dumps())
                writer.write(format_except(red.dumps()))






