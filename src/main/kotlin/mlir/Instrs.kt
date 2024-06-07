package me.alex_s168.uiua.mlir

object Inst {
    fun pDests(dests: List<MLIRVar>): String =
        if (dests.isEmpty()) "" else "${dests.joinToString()} = "

    fun undef(dest: MLIRVar, type: MLIRType) =
        "$dest = llvm.mlir.undef : $type"

    fun insertValue(dest: MLIRVar, structType: MLIRType, value: MLIRVar, into: MLIRVar, idx: Int) =
        "$dest = llvm.insertvalue $value, $into [$idx] : $structType"

    fun extractValue(dest: MLIRVar, structType: MLIRType, value: MLIRVar, idx: Int) =
        "$dest = llvm.extractvalue $value [$idx] : $structType"

    fun funcConstant(dest: MLIRVar, fn: String, fnType: MLIRType) =
        "$dest = func.constant $fn : $fnType"

    fun funcCallIndirect(dests: List<MLIRVar>, fn: MLIRVar, fnType: MLIRType, args: List<MLIRVar>) =
        "${pDests(dests)}func.call_indirect $fn(${args.joinToString()}) : $fnType"

    fun funcCall(dests: List<MLIRVar>, fn: String, fnType: MLIRType, vararg args: MLIRVar) =
        "${pDests(dests)}func.call @$fn(${args.joinToString()}) : $fnType"

    private fun baseBin(txt: String, dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        "$dest = " + (if (float) "arith.${txt}f $a, $b" else "arith.${txt}i $a, $b") + " : $type"

    fun add(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("add", dest, type, a, b, float)

    fun sub(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("sub", dest, type, a, b, float)

    fun mul(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("mul", dest, type, a, b, float)

    fun div(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("div", dest, type, a, b, float)

    private fun baseLinalg(fn: String, dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) =
        "linalg.$fn ins(${sources.joinToString { it.first }} : ${sources.joinToString { it.second }})\n  outs($dest : $destType)"

    fun arrAdd(dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) =
        baseLinalg("add", dest, destType, sources)

    fun arrSub(dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) =
        baseLinalg("sub", dest, destType, sources)

    fun arrMul(dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) =
        baseLinalg("mul", dest, destType, sources)

    fun arrDiv(dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) =
        baseLinalg("div", dest, destType, sources)

    fun tensorExtract(
        dest: MLIRVar,
        tensorType: MLIRType,
        tensor: MLIRVar,
        vararg idx: MLIRVar
    ) = "$dest = tensor.extract $tensor[${idx.joinToString()}] : $tensorType"

    fun tensorGenerate(
        dest: MLIRVar,
        tensorTy: MLIRType,
        dynamicDims: List<MLIRVar>,
        iterCords: List<MLIRVar>,
        iterRes: MLIRVar,
        iterResTy: MLIRType,
        inner: List<String>,
    ) = "$dest = tensor.generate ${dynamicDims.joinToString()} {\n^bb0(${iterCords.joinToString { "$it : index" }}):\n  ${inner.joinToString(separator = "\n  ")}\n  tensor.yield $iterRes : $iterResTy\n} : $tensorTy"

    fun tensorMemRef(dest: MLIRVar, tensorType: MLIRType, src: MLIRVar) =
        "$dest = bufferization.to_memref $src : $tensorType"

    fun memRefTensor(dest: MLIRVar, memrefType: MLIRType, src: MLIRVar) =
        "$dest = bufferization.to_tensor $src : $memrefType"

    fun tensorDim(dest: MLIRVar, tensorType: MLIRType, tensor: MLIRVar, dim: MLIRVar) =
        "$dest = tensor.dim $tensor, $dim : $tensorType"

    fun memRefDim(dest: MLIRVar, memRefType: MLIRType, memRef: MLIRVar, dim: MLIRVar) =
        "$dest = memref.dim $memRef, $dim : $memRefType"

    fun memRefLoad(
        dest: MLIRVar,
        memRefType: MLIRType,
        memRef: MLIRVar,
        vararg idx: MLIRVar
    ) = "$dest = memref.load $memRef[${idx.joinToString()}] : $memRefType"

    fun memRefStore(
        memRefType: MLIRType,
        value: MLIRVar,
        memRef: MLIRVar,
        vararg idx: MLIRVar
    ) = "memref.store $value, $memRef[${idx.joinToString()}] : $memRefType"

    fun memRefCopy(
        source: MLIRVar,
        sourceTy: MLIRType,
        dest: MLIRVar,
        destTy: MLIRType
    ) = "memref.copy $source, $dest : $sourceTy to $destTy"

    fun memRefAlloc(dest: MLIRVar, memRefType: MLIRType, vararg dim: MLIRVar) =
        "$dest = memref.alloc(${dim.joinToString()}) : $memRefType"

    fun memRefDealloc(memRef: MLIRVar, memRefType: MLIRType) =
        "memref.dealloc $memRef : $memRefType"

    fun affineParallelFor(counters: List<MLIRVar>, start: List<String>, end: List<String>, ops: List<String>) =
        "affine.parallel (${counters.joinToString()}) = (${start.joinToString()}) to (${end.joinToString()}) {\n  ${ops.joinToString("\n  ")}\n}"

    fun constant(dest: MLIRVar, type: MLIRType, value: String) =
        "$dest = arith.constant $value : $type"

    object Compound {
        fun memRefGenerate(
            dest: MLIRVar,
            memRefTy: MLIRType,
            allDims: List<MLIRVar>,
            dynDims: List<MLIRVar>,
            iterCords: List<MLIRVar>,
            iterRes: MLIRVar,
            iterResTy: MLIRType,
            inner: List<String>,
        ): List<String> {
            val res = mutableListOf<String>()
            res += memRefAlloc(dest, memRefTy, *dynDims.toTypedArray())
            val body = mutableListOf<String>()
            body.addAll(inner)
            body += memRefStore(memRefTy, iterRes, dest, *iterCords.toTypedArray())
            res += affineParallelFor(iterCords, allDims.map { "0" }, allDims, body)
            return res
        }
    }
}
