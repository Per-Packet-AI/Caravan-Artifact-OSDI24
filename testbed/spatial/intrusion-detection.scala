// *************************************************************************
//
// Copyright 2024 Qizheng Zhang (Stanford University),
//                Ali Imran (Purdue University),
//                Enkeleda Bardhi (Sapienza University of Rome),
//                Tushar Swamy (Unaffiliated),
//                Nathan Zhang (Stanford University),
//                Muhammad Shahbaz (Purdue University),
//                Kunle Olukotun (Stanford University)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// *************************************************************************

package spatial.tests.feature.transfers

import argon.static.Sym
import spatial.dsl._

@spatial class Main extends SpatialTest {
  import spatial.lang.{AxiStream512, AxiStream512Bus}
  // @struct case class AxiStream512(tdata: U512, tstrb: U64, tkeep: U64, tlast: Bit, tid: U8, tdest: U8, tuser: U64)
  def main(args: Array[String]): Unit = {
    // In/out buses here have type AxiStream512, meaning you can access all the axis fields in the Spatial source code (tdata, tstrb, tkeep, tlast, tid, tdest, tuser)
    //  If you only care about the tdata field, you should use type U512 instead of AxiStream512   
    type T  = FixPt[TRUE, _16, _16]    

    val x      = ArgIn[Int] // 08
    val y      = ArgIn[Int] // 0D
    
    val weights = List.tabulate(182){i => ArgIn[T]}

    val inbus = StreamIn[AxiStream512](AxiStream512Bus(tid = 0, tdest = 0))
    val outbus = StreamOut[AxiStream512](AxiStream512Bus(tid = 0, tdest = 1))
    
    Accel {
      val input = SRAM[T](16)
      val packet_use1 = FIFO[AxiStream512](2)
      val packet_use2 = FIFO[AxiStream512](10)
      val dummy_input_fifo = FIFO[T](2)
      val dummy_stage0_fifo = FIFO[T](2)
      val dummy_stage1_fifo = FIFO[T](2)
      val dummy_stage2_fifo = FIFO[T](2)
      val dummy_stage3_fifo = FIFO[T](2)
      
      val L0_res = SRAM[T](8)
      val L1_res = SRAM[T](4)
      val L2_res = SRAM[T](2)
      val L3_res = SRAM[T](2)

      val NUM_L0_R = 8 
      val NUM_L0_C = 16

      val L1_BASE  = (NUM_L0_R * NUM_L0_C) + NUM_L0_R

      val NUM_L1_R = 4
      val NUM_L1_C = 8

      val L2_BASE  = L1_BASE + (NUM_L1_R * NUM_L1_C) + NUM_L1_R
      
      val NUM_L2_R = 2
      val NUM_L2_C = 4
      
      val L0_W_LUT = SRAM[T](8,16)
      val L0_B_LUT = SRAM[T](8)
      val L1_W_LUT = SRAM[T](4,8)
      val L1_B_LUT = SRAM[T](4)
      val L2_W_LUT = SRAM[T](2,4)
      val L2_B_LUT = SRAM[T](2)
      val L3_W_LUT = SRAM[T](2,3)
      val L3_B_LUT = SRAM[T](2)

      Stream.Foreach(*) { stream_idx =>

        Pipe {
          val packet = inbus.value
          packet_use1.enq(packet)
          packet_use2.enq(packet)

          if(x === 1) {
            for(i <- 0 to (NUM_L0_R - 1)) {
              for(j <- 0 to (NUM_L0_C - 1)) {
                L0_W_LUT(i,j) = weights(j + (NUM_L0_C * i))
              }
            }

            for(j <- 0 to (NUM_L0_R - 1)) {
                L0_B_LUT(j)   = weights(j + (NUM_L0_R * NUM_L0_C))
            }

            for(i <- 0 to (NUM_L1_R - 1)) {
              for(j <- 0 to (NUM_L1_C - 1)) {
                L1_W_LUT(i,j) = weights(j + (NUM_L1_C * i) + L1_BASE)
              }
            }

            for(j <- 0 to (NUM_L1_R - 1)) {
                L1_B_LUT(j)   = weights(j + (NUM_L1_R * NUM_L1_C) + L1_BASE)
            }

            for(i <- 0 to (NUM_L2_R - 1)) {
              for(j <- 0 to (NUM_L2_C - 1)) {
                L2_W_LUT(i,j) = weights(j + (NUM_L2_C * i) + L2_BASE)
              }
            }

            for(j <- 0 to (NUM_L2_R - 1)) {
                L2_B_LUT(j)   = weights(j + (NUM_L2_R * NUM_L2_C) + L2_BASE)
            }
          }
        }


        Pipe {
          val packet = packet_use1.deq()
          val eth = 112
          val ip = 0
          val shift_amounts = Seq.tabulate(16){i => (eth + ip + (i * 16)).to[I16]}
          val zero = 0
          // val shift_amounts = Seq.tabulate(16){i => ((i * 32)).to[I16]}
          Foreach(0 until 16 par 16){ i =>
              val mux1H_conds = Seq.tabulate(16){j => j.to[I32] === i}
              val shifted_pkt = oneHotMux(mux1H_conds, shift_amounts.map{amt => packet.tdata.as[U512] >> amt})
              input(i) = cat(shifted_pkt.bits(7::7), shifted_pkt.bits(7::7), shifted_pkt.bits(7::7), shifted_pkt.bits(7::7), shifted_pkt.bits(7::7), shifted_pkt.bits(7::7), shifted_pkt.bits(7::7), shifted_pkt.bits(7::7) , shifted_pkt.bits(7::0), shifted_pkt.bits(15::8), zero.to[U512].bits(7::0) ).as[T]              
              // input(i) = cat(shifted_pkt.bits(7::0), shifted_pkt.bits(15::8)).as[T]//, shifted_pkt.bits(23::16), shifted_pkt.bits(31::24)).as[T]
          }
          dummy_input_fifo.enq(L1_res(2))
        }
        
        Pipe {
         
         val dummy = dummy_input_fifo.deq()
          
         List.tabulate(8) { i =>

              val partial_results = List.tabulate(16) { j =>
                  L0_W_LUT(i, j) * input(j)
              }

              val w = partial_results.reduceTree {_+_}
              L0_res(i) = max(w + L0_B_LUT(i), 0)
              // L0_res(i) = w + L0_B_LUT(i)

          }

          dummy_stage0_fifo.enq(L0_res(0))
        }
      
        Pipe {
          
          val dummy = dummy_stage0_fifo.deq()
        
          List.tabulate(4) { i =>
  
              val partial_results = List.tabulate(8) { j =>
                  L1_W_LUT(i, j) * L0_res(j)
              }

              val w = partial_results.reduceTree {_+_}
              L1_res(i) = max(w + L1_B_LUT(i), 0)

          }
          
          dummy_stage1_fifo.enq(L1_res(0))
        }
        

        Pipe {
          val dummy = dummy_stage1_fifo.deq()
        
          List.tabulate(2) { i =>

              val partial_results = List.tabulate(4) { j =>
                  L2_W_LUT(i, j) * L1_res(j)
              }

              val w = partial_results.reduceTree {_+_}
              L2_res(i) = w + L2_B_LUT(i)
        
          }
          
          dummy_stage2_fifo.enq(L2_res(0))
        }
              
        Pipe {
         val eth = 112
         val ip  = 0

         val dummy = dummy_stage2_fifo.deq()

         val decision = mux(L2_res(0) >= L2_res(1), 0, 1)
         
         val packet = packet_use2.deq()
         
         val newPacket = AxiStream512((packet.tdata.as[U512]) | (decision.as[U512] << (eth + ip + 16*16 + 8) ), packet.tstrb, packet.tkeep, packet.tlast, packet.tid, 1, 0)
         outbus := newPacket
        }

      }

    }

    assert(1 == 1) // Assert keeps spatial happy
  }
}
