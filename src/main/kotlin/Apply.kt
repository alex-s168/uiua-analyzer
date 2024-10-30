package me.alex_s168.uiua

import me.alex_s168.uiua.ir.*
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.system.measureTimeMillis

fun List<AnyPass>.apply(blocks: MutableMap<String, IrBlock>) {
    forEach {
        println("pass \"${it.name}\" started")

        val ti = measureTimeMillis {
            when (it) {
                is GlobalPass<*> -> {
                    it as GlobalPass<(IrBlock) -> Unit>
                    it.run(blocks, blocks::putBlock)
                }

                is Pass<*> -> {
                    it as Pass<(IrBlock) -> Unit>
                    if (it.parallel) {
                        val tpExec =
                            Executors.newScheduledThreadPool(Runtime.getRuntime().availableProcessors())
                        val exceptions = CopyOnWriteArrayList<Throwable>()
                        val new = CopyOnWriteArrayList<IrBlock>()
                        val flush = if (it.parallelDeepCopyBlocks) {
                            blocks.values.map { b ->
                                val f = b.deepCopy()
                                f.name = b.name
                                f.uid = b.uid
                                tpExec.execute {
                                    try {
                                        it.run(f) {
                                            new += it
                                        }
                                    } catch (e: Throwable) {
                                        exceptions += e
                                    }
                                }
                                f
                            }
                        } else {
                            blocks.values.forEach { b ->
                                tpExec.execute {
                                    try {
                                        it.run(b) {
                                            new += it
                                        }
                                    } catch (e: Throwable) {
                                        exceptions += e
                                    }
                                }
                            }
                            listOf()
                        }
                        tpExec.shutdown()
                        while (!tpExec.awaitTermination(100, TimeUnit.MILLISECONDS)) {
                            exceptions.forEach {
                                tpExec.shutdownNow()
                                throw it
                            }
                            exceptions.clear()
                        }
                        exceptions.forEach {
                            throw it
                        }
                        flush.forEach {
                            val old = blocks[it.name]!!
                            assert(it.name == old.name)
                            old.loadFrom(it)
                        }
                        new.forEach(blocks::putBlock)
                    } else {
                        val new = mutableListOf<IrBlock>()
                        blocks.values.forEach { b ->
                            it.run(b) {
                                new += it
                            }
                        }
                        new.forEach(blocks::putBlock)
                    }
                }
            }
        }

        println("pass \"${it.name}\" finished in ${ti}ms")
    }
}