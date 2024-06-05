package me.alex_s168.uiua.ir

import blitz.Provider
import me.alex_s168.uiua.*
import me.alex_s168.uiua.Function
import me.alex_s168.uiua.ast.ASTRoot
import me.alex_s168.uiua.ast.AstNode
import me.alex_s168.uiua.ast.astify
import kotlin.random.Random
import kotlin.random.nextULong

fun List<AstNode>.toIr(putAnonFn: (Function) -> String, blockArgs: List<IrVar>, instrs: MutableList<IrInstr>, newVar: Provider<IrVar>): List<IrVar> {
    val vars = List(size) { newVar() }

    forEachIndexed { index, node ->
        val variable = vars[index]

        node.value.getBBOrNull()?.let {
            val last = instrs.lastOrNull() ?: error("invalid use of extend")
            last.outs.add(variable)
        } ?: node.value.getBAOrNull()?.let {
            instrs.add(
                IrInstr(
                    outs = mutableListOf(variable),
                    instr = PrimitiveInstr("cUSE"),
                    args = mutableListOf(blockArgs[it.id]),
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
                            args = mutableListOf()
                        )
                    )
                }
                else -> {
                    val args = it.children.toIr(putAnonFn, blockArgs, instrs, newVar)
                    instrs.add(
                        IrInstr(
                            outs = mutableListOf(variable),
                            instr = it.instr,
                            args = args.toMutableList()
                        )
                    )
                }
            }
        }
    }

    return vars
}

fun Function.toIr(
    getFn: (String) -> IrBlock?,
    putFn: (IrBlock) -> Unit,
    name: String
): IrBlock {
    val ast = astify(children)
    require(ast.args == signature.inputs)
    val ir = ast.toIr(getFn, putFn, name)
    return ir
}

fun ASTRoot.toIr(
    getFn: (String) -> IrBlock?,
    putFn: (IrBlock) -> Unit,
    name: String
): IrBlock {
    fun anonFnName(): String =
        "_\$anon_${Random.nextULong()}"

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

    block.rets.addAll(0, children.toIr({
        anonFnName().also { name ->
            putFn(it.toIr(getFn, putFn, name))
        }
    }, block.args, block.instrs, block::newVar))

    return block
}

fun Map<String, Function>.toIr(): MutableMap<String, IrBlock> {
    val new = mutableMapOf<String, IrBlock>()

    forEach { (k, v) ->
        new.putBlock(v.toIr(new::get, new::putBlock, k))
    }

    return new
}