package me.alex_s168.uiua.ir

import blitz.Either
import blitz.Provider
import blitz.getBAOrNull
import blitz.getBBOrNull
import me.alex_s168.uiua.*
import me.alex_s168.uiua.Function
import me.alex_s168.uiua.ast.*

fun List<AstNode>.toIr(block: IrBlock, putAnonFn: (Function) -> String, blockArgs: List<IrVar>, instrs: MutableList<IrInstr>, newVar: Provider<IrVar>): List<IrVar> {
    val vars = List(size) { newVar() }

    forEachIndexed { index, node ->
        val variable = vars[index]

        node.value.getBBOrNull()?.let {
            val no = it.of
            val outIdx = it.resIdx
            block.instrs.filter { it.ast == no }.forEach { of ->
                repeat(outIdx + 1 - of.outs.size) {
                    of.outs.add(block.newVar())
                }

                block.instrs += IrInstr(
                    mutableListOf(variable),
                    PrimitiveInstr(Prim.Comp.USE),
                    mutableListOf(of.outs[outIdx]),
                )
            }
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
                    val resInst = when (it.instr) {
                        is PrimitiveInstr -> when (it.instr.id) {
                            Prim.IDENTITY -> PrimitiveInstr(Prim.Comp.USE)
                            else -> it.instr
                        }
                        else -> it.instr
                    }

                    val args = it.children.toIr(block, putAnonFn, blockArgs, instrs, newVar)
                    instrs.add(
                        IrInstr(
                            outs = mutableListOf(variable),
                            instr = resInst,
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
    name: String,
    astDest: MutableList<ASTRoot>
): IrBlock {
    val ast = astify(children)
    ast.functionName = name
    astDest += ast
    signature?.let {
        require(ast.args == it.inputs) {
            "mismatched signature  ast args: ${ast.args} ; signature args: ${signature.inputs}"
        }
    }

    val ir = ast.toIr(getFn, putFn, name, astDest)

    if (additionalDebugInstrs) {
        ir.instrs = ir.instrs.flatMapTo(mutableListOf()) { inst ->
            val idx = if (inst.instr is PrimitiveInstr) inst.instr.loc?.index
            else null

            idx?.let {
                listOf(
                    IrInstr(
                        mutableListOf(),
                        SourceLocInstr(it),
                        mutableListOf()
                    ),
                    inst
                )
            } ?: listOf(inst)
        }
    }

    return ir
}

fun ASTRoot.toIr(
    getFn: Map<String, IrBlock>,
    putFn: (IrBlock) -> Unit,
    name: String,
    astDest: MutableList<ASTRoot>
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

    val tbCorr = mutableListOf<Triple<AstNode, Int, IrVar>>()
    block.rets.addAll(0, children.toIr(block, {
        anonFnName().also { name ->
            it.value = Either.ofA(name)
            putFn(it.toIr(getFn, putFn, name, astDest))
        }
    }, block.args, block.instrs, block::newVar))

    tbCorr.forEach { (no, outIdx, variable) ->

    }

    return block
}

fun Map<String, Function>.toIr(astDest: MutableList<ASTRoot>): MutableMap<String, IrBlock> {
    val new = mutableMapOf<String, IrBlock>()

    forEach { (k, v) ->
        new.putBlock(v.toIr(new, new::putBlock, k, astDest))
    }

    return new
}