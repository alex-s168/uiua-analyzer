package me.alex_s168.uiua.mlir

import blitz.collections.contents
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrVar

fun castInstr(newVar: () -> IrVar, from: Type, to: Type, dest: IrVar, src: IrVar): List<String> {
    if (to != Types.dynamic && from == Types.dynamic) {
        val fn = UARuntime.dyn.castTo(to)
        return listOf(Inst.funcCall(
            dests = listOf(dest.asMLIR()),
            fn = fn.name,
            fnType = fn.type.toMLIR(),
            src.asMLIR()
        ))
    }

    return when (to) {
        Types.dynamic -> {
            val fn = UARuntime.dyn.createFrom(from)
            listOf(Inst.funcCall(
                listOf(dest.asMLIR()),
                fn.name,
                fn.type.toMLIR(),
                src.asMLIR(),
            ))
        }
        is ArrayType -> {
            require(from is ArrayType)
            // TODO: why tf does it even do implicit unbox
            if (from.shape.drop(1).contents == to.shape.contents) {
                val reac0 = List(from.shape.size-1){it.toString()}
                listOf(
                    "${dest.asMLIR()} = memref.collapse_shape ${src.asMLIR()} [${reac0.contents}, [${to.shape.size}]] : ${from.toMLIR()} into ${to.toMLIR()}"
                )
            }
            else {
                listOf(
                    "${dest.asMLIR()} = memref.cast ${src.asMLIR()} : ${from.toMLIR()} to ${to.toMLIR()}"
                )
            }
        }
        Types.double -> when (from) {
            Types.double -> error("No!")
            is PtrType -> error("double -> ptr not allowed")
            Types.int -> listOf("${dest.asMLIR()} = arith.sitofp ${src.asMLIR()} : i64 to f64")
            Types.size -> {
                val v = newVar().copy(type = Types.int).asMLIR()
                listOf(
                    "$v = index.castu ${src.asMLIR()} : index to i64",
                    "${dest.asMLIR()} = arith.uitofp $v : i64 to f64"
                )
            }
            Types.bool -> listOf("${dest.asMLIR()} = arith.uitofp ${src.asMLIR()} : i1 to f64")
            Types.byte,
            Types.autobyte -> listOf("${dest.asMLIR()} = arith.uitofp ${src.asMLIR()} : i8 to f64")
            else -> error("Cast from $from to $to not implemented")
        }
        is PtrType -> when (from) {
            Types.double -> error("ptr -> double not allowed")
            is PtrType -> error("No!")
            Types.int -> listOf("${dest.asMLIR()} = llvm.inttoptr ${src.asMLIR()} : i64 to !llvm.ptr")
            Types.size -> listOf("${dest.asMLIR()} = llvm.inttoptr ${src.asMLIR()} : index to !llvm.ptr")
            Types.byte,
            Types.autobyte -> listOf("${dest.asMLIR()} = llvm.inttoptr ${src.asMLIR()} : i8 to !llvm.ptr")
            Types.bool -> listOf("${dest.asMLIR()} = llvm.inttoptr ${src.asMLIR()} : i1 to !llvm.ptr")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.int -> when (from) {
            Types.double -> listOf("${dest.asMLIR()} = arith.fptosi ${src.asMLIR()} : f64 to i64")
            is PtrType -> listOf("${dest.asMLIR()} = llvm.ptrtoint ${src.asMLIR()} : !llvm.ptr to i64")
            Types.int -> error("No!")
            Types.size -> listOf("${dest.asMLIR()} = index.castu ${src.asMLIR()} : index to i64")
            Types.byte,
            Types.autobyte -> listOf("${dest.asMLIR()} = arith.extui ${src.asMLIR()} : i8 to i64")
            Types.bool -> listOf("${dest.asMLIR()} = arith.extui ${src.asMLIR()} : i1 to i64")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.byte -> when (from) {
            Types.double -> listOf("${dest.asMLIR()} = arith.fptoui ${src.asMLIR()} : f64 to i8")
            is PtrType -> listOf("${dest.asMLIR()} = llvm.ptrtoint ${src.asMLIR()} : !llvm.ptr to i8")
            Types.int -> listOf("${dest.asMLIR()} = arith.trunci ${src.asMLIR()} : i64 to i8")
            Types.size -> listOf("${dest.asMLIR()} = index.castu ${src.asMLIR()} : index to i8")
            Types.bool -> listOf("${dest.asMLIR()} = arith.extui ${src.asMLIR()} : i1 to i8")
            Types.byte,
            Types.autobyte -> error("No!")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.bool -> when (from) {
            Types.double -> listOf("${dest.asMLIR()} = arith.fptoui ${src.asMLIR()} : f64 to i1")
            is PtrType -> listOf("${dest.asMLIR()} = llvm.ptrtoint ${src.asMLIR()} : !llvm.ptr to i1")
            Types.int -> listOf("${dest.asMLIR()} = arith.trunci ${src.asMLIR()} : i64 to i1")
            Types.size -> listOf("${dest.asMLIR()} = index.castu ${src.asMLIR()} : index to i1")
            Types.byte,
            Types.autobyte -> listOf("${dest.asMLIR()} = arith.trunci ${src.asMLIR()} : i8 to i1")
            Types.bool -> error("no")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.size -> when (from) {
            Types.double -> {
                val v = newVar().copy(type = Types.int).asMLIR()
                listOf(
                    "$v = arith.fptoui ${src.asMLIR()} : f64 to i64",
                    "${dest.asMLIR()} = index.castu $v : i64 to index"
                )
            }
            Types.int -> listOf("${dest.asMLIR()} = index.castu ${src.asMLIR()} : i64 to index")
            Types.byte,
            Types.autobyte -> listOf("${dest.asMLIR()} = index.castu ${src.asMLIR()} : i8 to index")
            Types.bool -> listOf("${dest.asMLIR()} = index.castu ${src.asMLIR()} : i1 to index")
            is PtrType -> listOf("${dest.asMLIR()} = llvm.ptrtoint ${src.asMLIR()} : !llvm.ptr to index")
            Types.size -> error("No!")
            else -> error("Cast from $from to $to not implemented")
        }
        else -> error("Cast from $from to $to not implemented")
    }
}

fun castIfNec(newVar: () -> IrVar, body: MutableList<String>, variable: IrVar, want: Type): IrVar =
    if (want is ArrayType && variable.type !is ArrayType)
        castIfNec(newVar, body, variable, want.inner)
    else
        if (variable.type == want) variable
        else {
            val new = newVar().copy(type = want)
            body.addAll(castInstr(
                newVar,
                from = variable.type,
                to = want,
                dest = new,
                src = variable
            ))
            new
        }

fun castLaterIfNec(newVar: () -> IrVar, body: MutableList<String>, variable: IrVar, want: Type, block: (IrVar) -> Unit) {
    if (want is ArrayType && variable.type !is ArrayType)
        return castLaterIfNec(newVar, body, variable, want.inner, block)
    val dest = if (variable.type == want) variable
    else newVar().copy(type = want)
    block(dest)
    if (variable.type != want) {
        body.addAll(castInstr(
            newVar,
            from = want,
            to = variable.type,
            dest = variable,
            src = dest
        ))
    }
}