import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from data import PAD_IDX, SOS_IDX, EOS_IDX
DEBUG = False

def dprint(*args):
    if DEBUG:
        print(*args)

def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__( self, hidden_size, ):

        super(Attention, self).__init__()
        "Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)"
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward( self, query, encoder_outputs, src_lengths, ):
        dprint("\n#\n#\n#\n########## ATTENTION ######################\n#\n#\n#")
        # query: (batch_size, max_tgt_len, hidden_dim)
        # encoder_outputs: (batch_size, max_src_len, hidden_dim)
        # src_lengths: (batch_size)
        # we will need to use this mask to assign float("-inf") in the attention scores
        # of the padding tokens (such that the output of the softmax is 0 in those positions)
        # Tip: use torch.masked_fill to do this
        # src_seq_mask: (batch_size, max_src_len)
        # the "~" is the elementwise NOT operator
        src_seq_mask = ~self.sequence_mask(src_lengths)
        

        dprint("\n\n src_seq_mask.shape",src_seq_mask.shape)
        
        dprint("\n query.shape",query.shape)
        dprint("\n encode_outputs.shape",encoder_outputs.shape)
        
        # apply weights to query
        query_weighted = self.linear_in(query)
        dprint("\n query_weighted.shape",query_weighted.shape)

        score = torch.bmm(query_weighted,encoder_outputs.permute(0,2,1))
        dprint("\n score.shape",score.shape)
        
        # create mask for matrix instead of vector
        mask = src_seq_mask.unsqueeze(1).repeat(1,score.shape[1],1)
        dprint("\n mask.shape",mask.shape)
        
        
        score = score.masked_fill(mask,float("-inf"))

        # compute score vector        
        p = torch.softmax(score,dim=2)
        dprint("\n p.shape",p.shape)
        c = p @ encoder_outputs
        dprint("\n c.shape",c.shape)
        
        # compute attention layer output
        concat = torch.cat((query,c),dim=2) # [q;c]
        dprint("\n concat.shape",concat.shape)
        weighted_concat = self.linear_out(concat) # WoT [q;c]
        attn_out = torch.tanh(weighted_concat) # tanh(WoT[q;c])
        return attn_out
        #############################################
        # TODO: Implement the forward pass of the attention layer
        # Hints:
        # - Use torch.bmm to do the batch matrix multiplication
        #    (it does matrix multiplication for each sample in the batch)
        # - Use torch.softmax to do the softmax
        # - Use torch.tanh to do the tanh
        # - Use torch.masked_fill to do the masking of the padding tokens
        #############################################
        raise NotImplementedError
        #############################################
        # END OF YOUR CODE
        #############################################
        # attn_out: (batch_size, max_tgt_len, hidden_size)
        # TODO: Uncomment the following line when you implement the forward pass
        # return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(self,src_vocab_size, hidden_size, padding_idx,dropout,):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout
        self.embedding = nn.Embedding(src_vocab_size, hidden_size, padding_idx=padding_idx, )
        self.lstm = nn.LSTM(hidden_size, self.hidden_size, bidirectional=True, batch_first=True, )
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self,src,lengths,):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        dprint("enc forward",src.shape,"\n",src)
        dprint("\n\n",lengths)
        
        # do embedding of source sentence
        embedded = self.embedding(src)
        dprint("\n embeds: ", embedded.shape,"\n",embedded)
        
        # apply dropout
        drop_embedded = self.dropout(embedded)
        
        pack = True

        if pack:
        # pack embeddings to reduce computation cost
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(drop_embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        else:
            pack_embed = drop_embedded
        dprint("\n pack_embed: ",pack_embed)
        # run embeddings through lstm
        outputs,final_hidden = self.lstm(pack_embed)
        dprint("\n outputs :", outputs)
        
        dprint("\n state[0] shape:", final_hidden[0].shape)
        dprint("\n state[1] shape:", final_hidden[1].shape)
        
        # unpack results
        if pack:
            unpacked_outs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        else:
            unpacked_outs = outputs
        dprint("\n unpacked_outs: ", unpacked_outs.shape,"\n",unpacked_outs)
        # dprint("\n unpacked_lengths: ", unpacked_lengths.shape,"\n",unpacked_lengths)
        
        # dropout encoded outputs
        enc_output = self.dropout(unpacked_outs)
        dprint("\n final_hidden hash",hash(final_hidden))
        return enc_output, final_hidden

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors (state of the final layer of the lstm)
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)

class Decoder(nn.Module):
    def __init__( self, hidden_size, tgt_vocab_size, attn, padding_idx, dropout, ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding( self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM( self.hidden_size, self.hidden_size, batch_first=True, )

        self.attn = attn

    def forward( self, tgt, dec_state, encoder_outputs, src_lengths, ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)

        # reshape encoder state (bidirectional) to decoder state format (unidirectional)
        dprint("\ndec_state hash ori", hash(dec_state))
        if dec_state[0].shape[0] == 2:
            dprint("\n\nreshaped decoder")
            dec_state = reshape_state(dec_state)
            
        dprint("\n#\n#\n#\n########## DECODER ######################\n#\n#\n#")
        dprint("\n\n tgt.shape:",tgt.shape)
        dprint("\n\n tgt:",tgt.shape,"->",tgt)
        dprint("\n\n dec_state.shape:",(dec_state[0].shape,dec_state[1].shape))
        dprint("\n\n encoder_outputs.shape:",encoder_outputs.shape)
        # dprint("\n\n encoder_outputs:",encoder_outputs.shape,"->",encoder_outputs)
        dprint("\n\n src_lengths:",src_lengths.shape,"->",src_lengths)


        # remove EOS from target sentence
        if tgt.size(1) > 1:
            tgt_cln = tgt.clone()
            tgt_cln[tgt_cln == EOS_IDX] = PAD_IDX # remover EOS
            tgt_cut = tgt_cln[:,:-1]    # cut last word, since it will always be padding 
        else:
            tgt_cut = tgt
        dprint("\n\n tgt_cut:",tgt_cut.shape,"->",tgt_cut)
            
            
        # embed source
        emb_tgt = self.embedding(tgt_cut)

        # apply dropout
        drop_emb_tgt = self.dropout(emb_tgt)
        
        dprint("\ndrop_emb_tgt.shape",drop_emb_tgt.shape)
        # pass embedding through lstm
        outputs, dec_state = self.lstm(drop_emb_tgt,dec_state)
        dprint("\n outputs.shape",outputs.shape)
        
        # apply attention mechanism, if it is provided
        if self.attn is not None:
            outputs = self.attn( outputs, encoder_outputs, src_lengths, )

        # apply dropout
        outputs = self.dropout(outputs)

        return outputs, dec_state
        
        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)


class Seq2Seq(nn.Module):
    def __init__( self, encoder, decoder, ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)
        self.generator.weight = self.decoder.embedding.weight

    def forward( self, src, src_lengths, tgt, dec_hidden=None, ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder( tgt, dec_hidden, encoder_outputs, src_lengths )

        return self.generator(output), dec_hidden
