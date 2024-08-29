package me.alex_s168.uiua.ir

import blitz.Provider
import blitz.getBAOrNull
import blitz.getBBOrNull
import me.alex_s168.uiua.*
import me.alex_s168.uiua.Function
import me.alex_s168.uiua.ast.ASTRoot
import me.alex_s168.uiua.ast.AstNode
import me.alex_s168.uiua.ast.astify
import me.alex_s168.uiua.ast.printAst
import kotlin.random.Random
import kotlin.random.nextULong

fun List<AstNode>.toIr(tbCorr: MutableList<Pair<AstNode, IrVar>>, putAnonFn: (Function) -> String, blockArgs: List<IrVar>, instrs: MutableList<IrInstr>, newVar: Provider<IrVar>): List<IrVar> {
    val vars = List(size) { newVar() }

    forEachIndexed { index, node ->
        val variable = vars[index]

        node.value.getBBOrNull()?.let {
            tbCorr.add(it.of to variable)
        } ?: node.value.getBAOrNull()?.let {
            instrs.add(
                IrInstr(
                    outs = mutableListOf(variable),
                    instr = PrimitiveInstr("cUSE"),
                    args = mutableListOf(blockArgs[it.id]),
                    ast = node,
                )
            )
        } ?: node.value.getAOrNull()?.let {
            when (it.instr) {
                is PushFnInstr -> {
                    val anon = putAnonFn(it.instr.fn)
                    instrs.add(
                        IrInstr(
                            outs = mutableListOf(variable),
                            instr = PushFnRefInstr(anon),
                            args = mutableListOf(),
                            ast = node,
                        )
                    )
                }

                else -> {
                    val args = it.children.toIr(tbCorr, putAnonFn, blockArgs, instrs, newVar)
                    instrs.add(
                        IrInstr(
                            outs = mutableListOf(variable),
                            instr = it.instr,
                            args = args.toMutableList(),
                            ast = node,
                        )
                    )
                }
            }
        }
    }

    return vars
}

fun Function.toIr(
    getFn: Map<String, IrBlock>,
    putFn: (IrBlock) -> Unit,
    name: String
): IrBlock {
    val ast = astify(children)
    require(ast.args == signature.inputs) {
        "mismatched signature  ast args: ${ast.args} ; signature args: ${signature.inputs}"
    }

    val ir = ast.toIr(getFn, putFn, name)
    return ir
}

fun ASTRoot.toIr(
    getFn: Map<String, IrBlock>,
    putFn: (IrBlock) -> Unit,
    name: String
): IrBlock {
    val block = IrBlock(name, getFn)

    flagsAndComments.forEach {
        when (it) {
            is FlagInstr -> block.flags.add(it.flag)
        }
    }

    repeat(args) {
        val variable = block.newVar()
        block.args.add(variable)
    }

    val tbCorr = mutableListOf<Pair<AstNode, IrVar>>()
    block.rets.addAll(0, children.toIr(tbCorr, {
        anonFnName().also { name ->
            putFn(it.toIr(getFn, putFn, name))
        }
    }, block.args, block.instrs, block::newVar))

    tbCorr.forEach { (no, variable) ->
        val of = block.instrs.find { it.ast == no }!!
        of.outs.add(variable)
    }

    return block
}

fun Map<String, Function>.toIr(): MutableMap<String, IrBlock> {
    val new = mutableMapOf<String, IrBlock>()

    forEach { (k, v) ->
        new.putBlock(v.toIr(new, new::putBlock, k))
    }

    return new
}