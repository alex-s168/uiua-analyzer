package me.alex_s168.uiua.ir

import blitz.Either
import blitz.Provider
import blitz.getBAOrNull
import blitz.getBBOrNull
import me.alex_s168.uiua.*
import me.alex_s168.uiua.Function
import me.alex_s168.uiua.ast.*

fun List<AstNode>.toIr(tbCorr: MutableList<Triple<AstNode, Int, IrVar>>, putAnonFn: (Function) -> String, blockArgs: List<IrVar>, instrs: MutableList<IrInstr>, newVar: Provider<IrVar>): List<IrVar> {
    val vars = List(size) { newVar() }

    forEachIndexed { index, node ->
        val variable = vars[index]

        node.value.getBBOrNull()?.let {
            tbCorr.add(Triple(it.of, it.resIdx, variable))
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
    name: String,
    astDest: MutableList<ASTRoot>
): IrBlock {
    val ast = astify(children)
    ast.functionName = name
    astDest += ast
    require(ast.args == signature.inputs) {
        "mismatched signature  ast args: ${ast.args} ; signature args: ${signature.inputs}"
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
    block.rets.addAll(0, children.toIr(tbCorr, {
        anonFnName().also { name ->
            it.value = Either.ofA(name)
            putFn(it.toIr(getFn, putFn, name, astDest))
        }
    }, block.args, block.instrs, block::newVar))

    tbCorr.forEach { (no, outIdx, variable) ->
        val of = block.instrs.find { it.ast == no }!!

        repeat(outIdx + 1 - of.outs.size) {
            of.outs.add(block.newVar())
        }

        block.instrs += IrInstr(
            mutableListOf(variable),
            PrimitiveInstr(Prim.Comp.USE),
            mutableListOf(of.outs[outIdx]),
        )
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