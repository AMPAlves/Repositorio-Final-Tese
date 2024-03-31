% ====== Preferences ======

% @soft(aggregate) Prefer to use raw (no aggregate).
preference(aggregate,E) :-
    attribute((encoding,aggregate),E,_).

% @soft(bin) Prefer to not bin.
preference(bin,E) :-
    attribute((encoding,binning),E,_).

% @soft(bin_high) Prefer binning with at most 12 buckets.
preference(bin_high,E) :-
    attribute((encoding,binning),E,B), B > 12.

% @soft(bin_low) Prefer binning with more than 7 buckets.
preference(bin_low,E) :-
    attribute((encoding,binning),E,B), B <= 7.

% @soft(encoding) Prefer to use fewer encodings.
preference(encoding,E) :-
    entity(encoding,_,E).

% @soft(encoding_field) Prefer to use fewer encodings with fields (count does not have a field).
preference(encoding_field,E) :-
    attribute((encoding,field),E,_).

% @soft(same_field) Prefer not to use the same field twice for the same mark.
preference(same_field,F) :-
    entity(field,_,F),
    entity(mark,_,M),
    { entity(encoding,M,E): helper((encoding,field),E,F) } = 2.

% @soft(same_field_grt3) Prefer not to use the same field three or more times for the same mark.
preference(same_field_grt3,F) :-
    entity(field,_,F),
    entity(mark,_,M),
    { entity(encoding,M,E): helper((encoding,field),E,F) } >= 3.

% @soft(count_grt1) Prefer not to use count more than once.
preference(count_grt1,M) :-
    entity(mark,_,M),
    { entity(encoding,M,E): attribute((encoding,aggregate),E,count) } > 1.

% @soft(number_categorical) Should not use categorical scale for number field.
preference(number_categorical,E) :-
    attribute((field,type),F,number),
    helper((encoding,field),E,F),
    helper((encoding,scale_type),E,categorical).

% @soft(bin_low_unique) Binned field should not have too less unique values.
preference(bin_low_unique,E) :-
    attribute((field,type),F,(number;datetime)),
    attribute((field,unique),F,U),
    helper((encoding,field),E,F),
    attribute((encoding,binning),E,_),
    U < 15.

% @soft(bin_not_linear) Prefer linear scale for bin.
preference(bin_not_linear,E) :-
    attribute((encoding,binning),E,_),
    not helper((encoding,scale_type),E,linear).

% @soft(only_discrete) Only discrete encoding channels are used in a mark.
preference(only_discrete,M) :-
    entity(mark,_,M),
    not helper(mark_encoding_cont,M,_).

% @soft(multi_non_pos) Prefer not to use multiple non-positional encoding channels.
preference(multi_non_pos,M) :-
    entity(mark,_,M),
    { helper((mark,channel),M,C): domain(non_positional,C) } > 1.

% @soft(non_pos_used_before_pos) Prefer not to use non-positional channels until all positional channels are used.
preference(non_pos_used_before_pos,M) :-
    helper((mark,channel),M,C),
    domain(non_positional,C),
    not helper((mark,channel),M,(x;y)).

% @soft(aggregate_group_by_raw) Aggregate plots should not use raw continuous as group by.
preference(aggregate_group_by_raw,E) :-
    entity(encoding,M,EA),
    attribute((encoding,aggregate),EA,_),
    entity(encoding,M,E),
    not helper(mark_encoding_discrete_or_binned,M,E),
    not attribute((encoding,aggregate),E,_).

% @soft(aggregate_no_discrete) Aggregate should also have a discrete encoding to group by.
preference(aggregate_no_discrete,M) :-
    entity(encoding,M,EA),
    attribute((encoding,aggregate),EA,_),
    not helper(mark_encoding_discrete_or_binned,M,_).

% @soft(x_y_raw) Prefer not to use plot with both x and y discrete and no aggregate as it leads to occlusion.
preference(x_y_raw,M) :-
    helper(mark_channel_discrete_or_binned,M,x),
    helper(mark_channel_discrete_or_binned,M,y),
    entity(encoding,M,E),
    not helper(mark_encoding_discrete_or_binned,M,E),
    not attribute((encoding,aggregate),E,_).

% @soft(continuous_not_zero) Prefer to include zero for continuous (binned doesn't need zero).
preference(continuous_not_zero,E) :-
    not helper(mark_encoding_discrete_or_binned,M,E),
    helper(mark_encoding_scale,M,E,S),
    not attribute((scale,zero),S,true).

% @soft(size_not_zero) Prefer zero size (even when binned).
preference(size_not_zero,E) :-
    attribute((encoding,channel),E,size),
    helper(mark_encoding_scale,M,E,S),
    not attribute((scale,zero),S,true).

% @soft(continuous_pos_not_zero) Prefer zero continuous positional.
preference(continuous_pos_not_zero,E) :-
    attribute((encoding,channel),E,(x;y)),
    not helper(mark_encoding_discrete_or_binned,M,E),
    helper(mark_encoding_scale,M,E,S),
    not attribute((scale,zero),S,true).

% @soft(skew_zero) Prefer not to use zero when the difference between min and max is smaller than distance to 0.
preference(skew_zero,E) :-
    attribute((field,min),F,MIN),
    attribute((field,max),F,MAX),
    EXT = MAX - MIN,
    |MAX| > EXT,
    |MIN| > EXT,
    entity(encoding,M,E),
    helper((encoding,field),E,F),
    helper(mark_encoding_scale,M,E,S),
    attribute((scale,zero),S,true).

% @soft(cross_zero) Prefer not to include zero as baseline when the range of data crosses zero.
preference(cross_zero,E) :-
    attribute((field,min),F,MIN),
    attribute((field,max),F,MAX),
    MAX > 0,
    MIN < 0,
    entity(encoding,M,E),
    helper((encoding,field),E,F),
    helper(mark_encoding_scale,M,E,S),
    attribute((scale,zero),S,true).

% @soft(only_y) Prefer to use only x instead of only y.
preference(only_y,M) :-
    helper((mark,channel),M,y),
    not helper((mark,channel),M,x).

% @soft(only_y_relationship) Prefer to use only x instead of only y in the relationship task.
preference(only_y_relationship,M) :-
    attribute(task,root,relationship),
    helper((mark,channel),M,x),
    not helper((mark,channel),M,y).

% @soft(same_axis_multiple_marks) Prefer to only have a single mark for an axis.
preference(same_axis_multiple_marks,V) :-
    entity(mark,V,M1),
    entity(mark,V,M2),
    entity(encoding,M1,E1),
    entity(encoding,M2,E2),
    helper(same_axis,E1,E2).

% @soft(binning_zero_scale) Prefer to not zero the binning channel.
preference(binning_zero_scale,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(scale,V,S),
    entity(encoding,M,E),
    helper(binning_zero,E,S).

% @soft(binned_orientation_not_x) Prefer binned quantitative on x axis.
preference(binned_orientation_not_x,E) :-
    attribute((field,type),F,(number;datetime)),
    helper((encoding,field),E,F),
    attribute((encoding,channel),E,y),
    attribute((encoding,binning),E,_).

% @soft(high_cardinality_ordinal) Prefer not to use ordinal for fields with high cardinality.
preference(high_cardinality_ordinal,E) :-
    helper(encoding_cardinality,E,N),
    helper((encoding,scale_type),E,ordinal),
    N > 30.

% @soft(high_cardinality_categorical_grt10) Prefer not to use categorical (color) for fields with high cardinality.
preference(high_cardinality_categorical_grt10,E) :-
    helper(encoding_cardinality,E,N),
    helper((encoding,scale_type),E,categorical),
    N > 10.

% @soft(high_cardinality_shape) Prefer not to use high cardinality ordinal for shape.
preference(high_cardinality_shape,E) :-
    helper(encoding_cardinality,E,N),
    attribute((encoding,channel),E,shape),
    N > 8.

% @soft(high_cardinality_size) Prefer not to use size when the cardinality is large on x or y.
preference(high_cardinality_size,E) :-
    helper((mark,channel),M,size),
    entity(encoding,M,E),
    helper(encoding_cardinality,E,N),
    attribute((encoding,channel),E,(x;y)),
    not helper(mark_encoding_discrete_or_binned,M,E),
    N > 100.

% @soft(horizontal_scrolling_x) Avoid high cardinality on x as it causes horizontal scrolling.
preference(horizontal_scrolling_x,E) :-
    attribute((encoding,channel),E,x),
    helper(encoding_cardinality,E,N),
    helper(mark_encoding_discrete_or_binned,_,E),
    N > 50.

% @soft(horizontal_scrolling_col) Avoid high cardinality on column as it causes horizontal scrolling.
preference(horizontal_scrolling_col,F) :-
    attribute((facet,channel),F,col),
    helper(facet_cardinality,F,N),
    N > 5.

% @soft(date_scale) Prefer to use linear/ordinal scale type with dates.
preference(date_scale,E) :-
    attribute((field,type),F,datetime),
    helper((encoding,field),E,F),
    not helper((encoding,scale_type),E,linear),
    not helper((encoding,scale_type),E,ordinal).

% @soft(number_linear) Prefer use linear for numbers with high cardinality.
preference(number_linear,E) :-
    attribute((field,type),F,number),
    attribute((field,unique),F,N),
    N > 20,
    helper((encoding,field),E,F),
    not attribute((encoding,binning),E,_),
    not helper((encoding,scale_type),E,linear).

% @soft(position_entropy) Overplotting. Prefer not to use x and y for continuous with high cardinality and low entropy without aggregation because the points will overplot.
preference(position_entropy,E) :-
    attribute((encoding,channel),E,(x;y)),
    attribute(encoding_cardinality,E,N),
    N > 100,
    attribute((field,entropy),F,EN),
    helper((encoding,field),E,F),
    EN <= 3000,
    not helper(mark_encoding_discrete_or_binned,_,E),
    not attribute((encoding,aggregate),E,_).

% @soft(value_agg) Prefer not to aggregate for value tasks.
preference(value_agg,V) :-
    attribute(task,root,value),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,_).

% @soft(relationship_facet) Prefer not to use facet for relationship tasks as it makes it difficult to compare.
preference(relationship_facet,V) :-
    attribute(task,root,relationship),
    entity(facet,V,_).

% @soft(summary_facet) Prefer not to use facet for summary tasks as it makes it difficult to compare.
preference(summary_facet,V) :-
    attribute(task,root,summary),
    entity(facet,V,_).

% @soft(c_d_col) Prefer not to use continuous on x, discrete on y, and column.
preference(c_d_col,V) :-
    entity(mark,V,M),
    not helper(mark_channel_discrete_or_binned,M,x),
    helper(mark_channel_discrete_or_binned,M,y),
    entity(facet,V,F),
    attribute((facet,channel),F,col).

% @soft(date_not_x) Prefer datetime on x.
preference(date_not_x,E) :-
    attribute((field,type),F,datetime),
    helper((encoding,field),E,F),
    not attribute((encoding,channel),E,x).

% @soft(x_row) Positional interactions as suggested by Kim et al.
preference(x_row,V) :-
    entity(mark,V,M),
    helper((mark,channel),M,x),
    entity(facet,V,F),
    attribute((facet,channel),F,row).

% @soft(y_row) Positional interactions as suggested by Kim et al.
preference(y_row,V) :-
    entity(mark,V,M),
    helper((mark,channel),M,y),
    entity(facet,V,F),
    attribute((facet,channel),F,row).

% @soft(x_col) Positional interactions as suggested by Kim et al.
preference(x_col,V) :-
    entity(mark,V,M),
    helper((mark,channel),M,x),
    entity(facet,V,F),
    attribute((facet,channel),F,col).

% @soft(y_col) Positional interactions as suggested by Kim et al.
preference(y_col,V) :-
    entity(mark,V,M),
    helper((mark,channel),M,y),
    entity(facet,V,F),
    attribute((facet,channel),F,col).

% @soft(color_entropy_high) Entropy, primary quantive interactions as suggested by Kim et al.
preference(color_entropy_high,E) :-
    attribute((field,entropy),F,EN),
    helper((encoding,field),E,F),
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,linear),
    helper(enc_interesting,E),
    EN > 3000.

% @soft(color_entropy_low) Entropy, primary quantive interactions as suggested by Kim et al.
preference(color_entropy_low,E) :-
    attribute((field,entropy),F,EN),
    helper((encoding,field),E,F),
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,linear),
    helper(enc_interesting,E),
    EN <= 3000.

% @soft(size_entropy_high) Entropy, primary quantive interactions as suggested by Kim et al.
preference(size_entropy_high,E) :-
    attribute((field,entropy),F,EN),
    helper((encoding,field),E,F),
    attribute((encoding,channel),E,size),
    helper((encoding,scale_type),E,linear),
    helper(enc_interesting,E),
    EN > 3000.

% @soft(size_entropy_low) Entropy, primary quantive interactions as suggested by Kim et al.
preference(size_entropy_low,E) :-
    attribute((field,entropy),F,EN),
    helper((encoding,field),E,F),
    attribute((encoding,channel),E,size),
    helper((encoding,scale_type),E,linear),
    helper(enc_interesting,E),
    EN <= 3000.

% @soft(linear_scale) linear scale.
preference(linear_scale,E) :-
    helper((encoding,scale_type),E,linear).

% @soft(log_scale) log scale.
preference(log_scale,E) :-
    helper((encoding,scale_type),E,log).

% @soft(ordinal_scale) ordinal scale.
preference(ordinal_scale,E) :-
    helper((encoding,scale_type),E,ordinal).

% @soft(categorical_scale) categorical scale.
preference(categorical_scale,E) :-
    helper((encoding,scale_type),E,categorical).

% @soft(c_c_point) Continuous by continuous for point mark.
preference(c_c_point,M) :-
    helper(is_c_c,M),
    attribute((mark,type),M,point).

% @soft(c_c_line) Continuous by continuous for line mark.
preference(c_c_line,M) :-
    helper(is_c_c,M),
    attribute((mark,type),M,line).

% @soft(c_c_area) Continuous by continuous for area mark.
preference(c_c_area,M) :-
    helper(is_c_c,M),
    attribute((mark,type),M,area).

% @soft(c_c_text) Continuous by continuous for text mark.
preference(c_c_text,M) :-
    helper(is_c_c,M),
    attribute((mark,type),M,text).

% @soft(c_d_overlap_point) Continuous by discrete for point mark.
preference(c_d_overlap_point,M) :-
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,point).

% @soft(c_d_overlap_bar) Continuous by discrete for bar mark.
preference(c_d_overlap_bar,M) :-
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,bar).

% @soft(c_d_overlap_line) Continuous by discrete for line mark.
preference(c_d_overlap_line,M) :-
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,line),
    not attribute(task,root,_).

% @soft(c_d_overlap_area) Continuous by discrete for area mark.
preference(c_d_overlap_area,M) :-
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,area).

% @soft(c_d_overlap_text) Continuous by discrete for text mark.
preference(c_d_overlap_text,M) :-
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,text).

% @soft(c_d_overlap_tick) Continuous by discrete for tick mark.
preference(c_d_overlap_tick,M) :-
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,tick).

% @soft(c_d_no_overlap_point) Continuous by discrete for point mark.
preference(c_d_no_overlap_point,M) :-
    helper(is_c_d,M),
    helper(no_overlap,M),
    attribute((mark,type),M,point).

% @soft(c_d_no_overlap_bar) Continuous by discrete for bar mark.
preference(c_d_no_overlap_bar,M) :-
    helper(is_c_d,M),
    helper(no_overlap,M),
    attribute((mark,type),M,bar).

% @soft(c_d_no_overlap_line) Continuous by discrete for line mark.
preference(c_d_no_overlap_line,M) :-
    helper(is_c_d,M),
    helper(no_overlap,M),
    attribute((mark,type),M,line).

% @soft(c_d_no_overlap_area) Continuous by discrete for area mark.
preference(c_d_no_overlap_area,M) :-
    helper(is_c_d,M),
    helper(no_overlap,M),
    attribute((mark,type),M,area).

% @soft(c_d_no_overlap_text) Continuous by discrete for text mark.
preference(c_d_no_overlap_text,M) :-
    helper(is_c_d,M),
    helper(no_overlap,M),
    attribute((mark,type),M,text).

% @soft(c_d_no_overlap_tick) Continuous by discrete for tick mark.
preference(c_d_no_overlap_tick,M) :-
    helper(is_c_d,M),
    helper(no_overlap,M),
    attribute((mark,type),M,tick).

% @soft(d_d_overlap) Prefer not to overlap with DxD.
preference(d_d_overlap,M) :-
    helper(is_d_d,M),
    helper(overlap,M).

% @soft(d_d_point) Discrete by discrete for point mark.
preference(d_d_point,M) :-
    helper(is_d_d,M),
    attribute((mark,type),M,point).

% @soft(d_d_text) Discrete by discrete for text mark.
preference(d_d_text,M) :-
    helper(is_d_d,M),
    attribute((mark,type),M,text).

% @soft(d_d_rect) Discrete by discrete for rect mark.
preference(d_d_rect,M) :-
    helper(is_d_d,M),
    attribute((mark,type),M,rect).

% @soft(linear_x) Linear scale with x channel.
preference(linear_x,E) :-
    attribute((encoding,channel),E,x),
    helper((encoding,scale_type),E,linear).

% @soft(linear_y) Linear scale with y channel.
preference(linear_y,E) :-
    attribute((encoding,channel),E,y),
    helper((encoding,scale_type),E,linear).

% @soft(linear_color) Linear scale with color channel.
preference(linear_color,E) :-
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,linear).

% @soft(linear_size) Linear scale with size channel.
preference(linear_size,E) :-
    attribute((encoding,channel),E,size),
    helper((encoding,scale_type),E,linear).

% @soft(linear_text) Linear scale with text channel.
preference(linear_text,E) :-
    attribute((encoding,channel),E,text),
    helper((encoding,scale_type),E,linear).

% @soft(log_x) Log scale with x channel.
preference(log_x,E) :-
    attribute((encoding,channel),E,x),
    helper((encoding,scale_type),E,log).

% @soft(log_y) Log scale with y channel.
preference(log_y,E) :-
    attribute((encoding,channel),E,y),
    helper((encoding,scale_type),E,log).

% @soft(log_color) Log scale with color channel.
preference(log_color,E) :-
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,log).

% @soft(log_size) Log scale with size channel.
preference(log_size,E) :-
    attribute((encoding,channel),E,size),
    helper((encoding,scale_type),E,log).

% @soft(log_text) Log scale with text channel.
preference(log_text,E) :-
    attribute((encoding,channel),E,text),
    helper((encoding,scale_type),E,log).

% @soft(ordinal_x) Ordinal scale with x channel.
preference(ordinal_x,E) :-
    attribute((encoding,channel),E,x),
    helper((encoding,scale_type),E,ordinal).

% @soft(ordinal_y) Ordinal scale with y channel.
preference(ordinal_y,E) :-
    attribute((encoding,channel),E,y),
    helper((encoding,scale_type),E,ordinal).

% @soft(ordinal_color) Ordinal scale with color channel.
preference(ordinal_color,E) :-
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,ordinal).

% @soft(ordinal_size) Ordinal scale with size channel.
preference(ordinal_size,E) :-
    attribute((encoding,channel),E,size),
    helper((encoding,scale_type),E,ordinal).

% @soft(ordinal_shape) Ordinal scale with shape channel.
preference(ordinal_shape,E) :-
    attribute((encoding,channel),E,shape),
    helper((encoding,scale_type),E,ordinal).

% @soft(ordinal_text) Ordinal scale with text channel.
preference(ordinal_text,E) :-
    attribute((encoding,channel),E,text),
    helper((encoding,scale_type),E,ordinal).

% @soft(ordinal_detail) Ordinal scale with detail channel.
preference(ordinal_detail,E) :-
    attribute((encoding,channel),E,detail),
    helper((encoding,scale_type),E,ordinal).



% @soft(numcate_x) numcate scale with x channel.
preference(numcate_x,E) :-
    attribute((encoding,channel),E,x),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_y) numcate scale with y channel.
preference(numcate_y,E) :-
    attribute((encoding,channel),E,y),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_color) numcate scale with color channel.
preference(numcate_color,E) :-
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_size) numcate scale with size channel.
preference(numcate_size,E) :-
    attribute((encoding,channel),E,size),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_shape) numcate scale with shape channel.
preference(numcate_shape,E) :-
    attribute((encoding,channel),E,shape),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_text) numcate scale with text channel.
preference(numcate_text,E) :-
    attribute((encoding,channel),E,text),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_detail) numcate scale with detail channel.
preference(numcate_detail,E) :-
    attribute((encoding,channel),E,detail),
    helper((encoding,scale_type),E,numcate).

% @soft(numcate_scale) numcate scale.
preference(numcate_scale,E) :-
    helper((encoding,scale_type),E,numcate).

% @soft(categorical_color) Categorical scale with color channel.
preference(categorical_color,E) :-
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,categorical).

% @soft(aggregate_count) Count as aggregate op.
preference(aggregate_count,E) :-
    attribute((encoding,aggregate),E,count).

% @soft(aggregate_mean) Mean as aggregate op.
preference(aggregate_mean,E) :-
    attribute((encoding,aggregate),E,mean).

% @soft(aggregate_median) Median as aggregate op.
preference(aggregate_median,E) :-
    attribute((encoding,aggregate),E,median).

% @soft(aggregate_min) Min as aggregate op.
preference(aggregate_min,E) :-
    attribute((encoding,aggregate),E,min).

% @soft(aggregate_max) Max as aggregate op.
preference(aggregate_max,E) :-
    attribute((encoding,aggregate),E,max).

% @soft(aggregate_stdev) Stdev as aggregate op.
preference(aggregate_stdev,E) :-
    attribute((encoding,aggregate),E,stdev).

% @soft(aggregate_sum) Sum as aggregate op.
preference(aggregate_sum,E) :-
    attribute((encoding,aggregate),E,sum).

% @soft(stack_zero) Zero base for stack op.
preference(stack_zero,E) :-
    attribute((encoding,stack),E,zero).

% @soft(stack_center) Center groupbys as stack op.
preference(stack_center,E) :-
    attribute((encoding,stack),E,center).

% @soft(stack_normalize) Normalize between groupbys as stack op.
preference(stack_normalize,E) :-
    attribute((encoding,stack),E,normalize).

% @soft(categoricalcolorcardinality) Penalise categorical with a cardinality over 11 (since there are only 11 colors).
preference(categoricalcolorcardinality,F) :-
    attribute((field,name),F,N),
    attribute((field,unique),F,U),
    attribute((encoding,field),E,N),
    attribute((encoding,channel),E,color),
    helper((encoding,scale_type),E,categorical),
    U >= 11.

% @soft(area_without_aggregate) Areas are prefered with aggregate.
preference(area_without_aggregate,M) :-
    attribute((mark,type),M,area),
    not attribute((encoding,aggregate),_,_).

% @soft(cdd_not_all_fields) Facetting is a better outcome when analysing CxDxD variables.
preference(cdd_not_all_fields,E) :-
    entity(field,_,A),
    entity(field,_,B),
    entity(field,_,C),
    entity(encoding,_,E),
    attribute((field,name),A,N1),
    attribute((field,type),A,(number;datetime)),
    attribute((field,name),B,N2),
    attribute((field,type),B,(string;boolean)),
    attribute((field,name),C,N3),
    attribute((field,type),C,(string;boolean)),
    attribute((encoding,field),_,N1),
    attribute((encoding,field),_,N2),
    not attribute((encoding,field),_,N3).

% @soft(cdd_facet) Facetting is a better outcome when analysing CxDxD variables.
preference(cdd_facet,F) :-
    entity(field,_,A),
    entity(field,_,B),
    entity(field,_,C),
    entity(facet,_,F),
    attribute((field,name),A,N1),
    attribute((field,type),A,(number;datetime)),
    attribute((field,name),B,N2),
    attribute((field,type),B,(string;boolean)),
    attribute((field,name),C,N3),
    attribute((field,type),C,(string;boolean)),
    attribute((encoding,field),_,N1),
    attribute((encoding,field),_,N2),
    not attribute((encoding,field),_,N3),
    not attribute((facet,field),_,N3).

% @soft(size_with_agg) Size should have an aggregate.
preference(size_with_agg,E) :-
    entity(encoding,_,E),
    attribute((encoding,channel),E,size),
    not attribute((encoding,aggregate),E,sum).

% @soft(relationship_log_scale) log scale.
preference(relationship_log_scale,E) :-
    attribute(task,root,relationship),
    helper((encoding,scale_type),E,log).

% @soft(relationship_binning) Prefer not to bin for relationship tasks.
preference(relationship_binning,E) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,binning),E,_).

% @soft(relationship_count) Prefer not to aggregate (count) for relationship tasks.
preference(relationship_count,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,count).

% @soft(relationship_median) Prefer not to aggregate (median) for relationship tasks.
preference(relationship_median,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,median).

% @soft(relationship_stdev) Prefer not to aggregate (stdev) for relationship tasks.
preference(relationship_stdev,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,stdev).

% @soft(relationship_min) Prefer not to aggregate (min) for relationship tasks.
preference(relationship_min,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,min).

% @soft(relationship_max) Prefer not to aggregate (max) for relationship tasks.
preference(relationship_max,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,max).

% @soft(relationship_sum) Prefer to aggregate (sum) for relationship tasks.
preference(relationship_sum,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,sum).

% @soft(relationship_mean) Prefer to aggregate (mean) for relationship tasks.
preference(relationship_mean,V) :-
    attribute(task,root,relationship),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,mean).

% @soft(relationship_point) Point mark for relationship tasks.
preference(relationship_point,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,point).

% @soft(relationship_bar) Bar mark for relationship tasks.
preference(relationship_bar,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,bar).

% @soft(relationship_line) Line mark for relationship tasks.
preference(relationship_line,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,line).

% @soft(relationship_area) Area mark for relationship tasks.
preference(relationship_area,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,area).

% @soft(relationship_text) Text mark for relationship tasks.
preference(relationship_text,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,text).

% @soft(relationship_tick) Tick mark for relationship tasks.
preference(relationship_tick,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,tick).

% @soft(relationship_rect) Rect mark for relationship tasks.
preference(relationship_rect,M) :-
    attribute(task,root,relationship),
    attribute((mark,type),M,rect).

% @soft(relationship_cd_overlap_line) Continuous by discrete for line mark in Relationship tasks.
preference(relationship_cd_overlap_line,M) :-
    attribute(task,root,relationship),
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,line).

% @soft(distribution_log_scale) log scale.
preference(distribution_log_scale,E) :-
    attribute(task,root,distribution),
    helper((encoding,scale_type),E,log).

% @soft(distribution_binning) Prefer not to bin for distribution tasks.
preference(distribution_binning,E) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,binning),E,_).

% @soft(distribution_count) Prefer not to aggregate (count) for distribution tasks.
preference(distribution_count,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,count).

% @soft(distribution_median) Prefer not to aggregate (median) for distribution tasks.
preference(distribution_median,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,median).

% @soft(distribution_stdev) Prefer not to aggregate (stdev) for distribution tasks.
preference(distribution_stdev,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,stdev).

% @soft(distribution_min) Prefer not to aggregate (min) for distribution tasks.
preference(distribution_min,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,min).

% @soft(distribution_max) Prefer not to aggregate (max) for distribution tasks.
preference(distribution_max,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,max).

% @soft(distribution_sum) Prefer to aggregate (sum) for distribution tasks.
preference(distribution_sum,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,sum).

% @soft(distribution_mean) Prefer to aggregate (mean) for distribution tasks.
preference(distribution_mean,V) :-
    attribute(task,root,distribution),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,mean).

% @soft(distribution_point) Point mark for distribution tasks.
preference(distribution_point,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,point).

% @soft(distribution_bar) Bar mark for distribution tasks.
preference(distribution_bar,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,bar).

% @soft(distribution_line) Line mark for distribution tasks.
preference(distribution_line,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,line).

% @soft(distribution_area) Area mark for distribution tasks.
preference(distribution_area,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,area).

% @soft(distribution_text) Text mark for distribution tasks.
preference(distribution_text,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,text).

% @soft(distribution_tick) Tick mark for distribution tasks.
preference(distribution_tick,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,tick).

% @soft(distribution_rect) Rect mark for distribution tasks.
preference(distribution_rect,M) :-
    attribute(task,root,distribution),
    attribute((mark,type),M,rect).

% @soft(distribution_not_single_count) Preference for count aggregate in single variables.
preference(distribution_not_single_count,F) :-
    attribute(task,root,distribution),
    attribute((field,name),F,N),
    attribute((field,type),F,(number;datetime;string)),
    entity(encoding,_,E),
    attribute((encoding,field),E,N),
    not attribute((encoding,aggregate),_,count).

% @soft(distribution_single_aggregate) Preference for having a single aggregate per chart.
preference(distribution_single_aggregate,E1) :-
    attribute(task,root,distribution),
    entity(encoding,_,E2),
    not helper(same_axis,E1,E2),
    attribute((encoding,aggregate),E1,count),
    attribute((encoding,aggregate),E2,_),
    E1 != E2.

% @soft(distribution_discrete_x) Preference for discrete on the x axis.
preference(distribution_discrete_x,F) :-
    attribute(task,root,distribution),
    attribute((field,name),F,N),
    attribute((field,type),F,string),
    entity(encoding,_,E),
    attribute((encoding,field),E,N),
    not attribute((encoding,channel),E,x).

% @soft(distribution_polar_coordinate) Polar coordinates.
preference(distribution_polar_coordinate,V) :-
    attribute(task,root,distribution),
    attribute((view,coordinates),V,polar).

% @soft(distribution_c_c_facet) Facet are not great in a continuous_continuous
preference(distribution_c_c_facet,F) :-
    attribute(task,root,distribution),
    entity(field,_,F1),
    entity(field,_,F2),
    attribute((field,name),F1,A),
    attribute((field,type),F1,(number;datetime)),
    attribute((field,name),F2,B),
    attribute((field,type),F2,(number;datetime)),
    entity(encoding,_,E1),
    entity(encoding,_,E2),
    attribute((encoding,field),E1,A),
    attribute((encoding,field),E2,B),
    attribute((encoding,channel),E1,x),
    attribute((encoding,channel),E2,y),
    attribute((facet,field),F,(A;B)).

% @soft(distribution_c_bin_y) Distribution with a single variable favors log and bin.
preference(distribution_c_bin_y,F) :-
    attribute(task,root,distribution),
    entity(mark,_,M),
    attribute((field,name),F,A),
    attribute((field,type),F,number),
    attribute((encoding,field),E,A),
    attribute((encoding,channel),E,y),
    not attribute((encoding,channel),_,x),
    not attribute((encoding,binning),E,(10;25;200)).

% @soft(distribution_c_bin_x) Distribution with a single variable favors log and bin.
preference(distribution_c_bin_x,F) :-
    attribute(task,root,distribution),
    entity(mark,_,M),
    attribute((field,name),F,A),
    attribute((field,type),F,number),
    attribute((encoding,field),E,A),
    attribute((encoding,channel),E,x),
    not attribute((encoding,channel),_,y),
    not attribute((encoding,binning),E,(10;25;200)).

% @soft(distribution_bin_without_log) Bins are bettwer visualiazed with a log.
preference(distribution_bin_without_log,F) :-
    attribute(task,root,distribution),
    entity(encoding,_,E),
    entity(scale,_,S),
    attribute((field,name),F,A),
    attribute((encoding,field),E,A),
    attribute((encoding,binning),E,_),
    not attribute((scale,type),S,log).

% @soft(count_without_log) Count should not typically have log.
preference(count_without_log,E) :-
    attribute(task,root,distribution),
    entity(encoding,_,E),
    entity(scale,_,S),
    attribute((encoding,aggregate),E,count),
    attribute((scale,type),S,log).

% @soft(distribution_bin_200) Distribution prefers 25 or 10 bin.
preference(distribution_bin_200,E) :-
    attribute(task,root,distribution),
    attribute((encoding,binning),E,200).

% @soft(distribution_bin_25) Distribution prefers 10 or 200 bin.
preference(distribution_bin_25,E) :-
    attribute(task,root,distribution),
    attribute((encoding,binning),E,25).

% @soft(distribution_bin_10) Distribution prefers 25 or 200 bin.
preference(distribution_bin_10,E) :-
    attribute(task,root,distribution),
    attribute((encoding,binning),E,10).

% @soft(distribution_x_y_raw) Prefer not to use plot with both x and y discrete and no aggregate as it leads to occlusion.
preference(distribution_x_y_raw,M) :-
    attribute(task,root,distribution),
    helper(mark_channel_discrete_or_binned,M,x),
    helper(mark_channel_discrete_or_binned,M,y),
    entity(encoding,M,E),
    not helper(mark_encoding_discrete_or_binned,M,E),
    not attribute((encoding,aggregate),E,_).

% @soft(distribution_cd_line_bar_aggregate) Lines with countinuous and discrete prefer aggregate.
preference(distribution_cd_line_bar_aggregate,M) :-
    attribute(task,root,distribution),
    entity(mark,_,M),
    attribute((mark,type),M,(line;bar)),
    not attribute((encoding,aggregate),_,_),
    { entity(encoding,M,E): attribute((encoding,field),E,F) } = 2.

% @soft(distribution_continuous_pos_not_zero) Prefer zero continuous positional.
preference(distribution_continuous_pos_not_zero,E) :-
    attribute(task,root,distribution),
    attribute((encoding,channel),E,(x;y)),
    not helper(mark_encoding_discrete_or_binned,M,E),
    helper(mark_encoding_scale,M,E,S),
    not attribute((scale,zero),S,true).

% @soft(distribution_cd_overlap_line) Continuous by discrete for line mark in Distribution tasks.
preference(distribution_cd_overlap_line,M) :-
    attribute(task,root,distribution),
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,line).

% @soft(compare_binning) Prefer not to bin for compare tasks.
preference(compare_binning,E) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    not attribute((encoding,binning),E,(10;25;200)).

% @soft(compare_bin_ordinal_y) Prefer not to have ordinal in the y axis.
preference(compare_bin_ordinal_y,E) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    entity(scale,V,S),
    attribute((encoding,channel),E,x),
    attribute((encoding,binning),E,_),
    attribute((scale,channel),S,x),
    not attribute((scale,type),S,ordinal).

% @soft(compare_count) Prefer not to aggregate (count) for compare tasks.
preference(compare_count,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,count).

% @soft(compare_median) Prefer not to aggregate (median) for compare tasks.
preference(compare_median,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,median).

% @soft(compare_stdev) Prefer not to aggregate (stdev) for compare tasks.
preference(compare_stdev,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,stdev).

% @soft(compare_min) Prefer not to aggregate (min) for compare tasks.
preference(compare_min,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,min).

% @soft(compare_max) Prefer not to aggregate (max) for compare tasks.
preference(compare_max,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,max).

% @soft(compare_sum) Prefer to aggregate (sum) for compare tasks.
preference(compare_sum,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,sum).

% @soft(compare_mean) Prefer to aggregate (mean) for compare tasks.
preference(compare_mean,V) :-
    attribute(task,root,compare),
    entity(mark,V,M),
    entity(encoding,M,E),
    attribute((encoding,aggregate),E,mean).

% @soft(compare_point) Point mark for compare tasks.
preference(compare_point,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,point).

% @soft(compare_bar) Bar mark for compare tasks.
preference(compare_bar,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,bar).

% @soft(compare_line) Line mark for compare tasks.
preference(compare_line,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,line).

% @soft(compare_area) Area mark for compare tasks.
preference(compare_area,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,area).

% @soft(compare_text) Text mark for compare tasks.
preference(compare_text,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,text).

% @soft(compare_tick) Tick mark for compare tasks.
preference(compare_tick,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,tick).

% @soft(compare_rect) Rect mark for compare tasks.
preference(compare_rect,M) :-
    attribute(task,root,compare),
    attribute((mark,type),M,rect).

% @soft(compare_c_c_point) Continuous by continuous for point mark.
preference(compare_c_c_point,M) :-
    attribute(task,root,compare),
    helper(is_c_c,M),
    attribute((mark,type),M,point).

% @soft(compare_continuous_pos_not_zero) Prefer zero continuous positional.
preference(compare_continuous_pos_not_zero,M) :-
    attribute(task,root,(compare;relationship;distribution)),
    entity(view,root,_),
    attribute((view,coordinates),_,cartesian),
    entity(encoding,M,E1),
    entity(encoding,M,E2),
    attribute((encoding,channel),E1,x),
    attribute((encoding,channel),E2,y),
    attribute((scale,channel),S1,x),
    attribute((scale,channel),S2,y),
    attribute((scale,zero),S1,true),
    not attribute((scale,zero),S2,true),
    E1 != E2,
    S1 != S2.

% @soft(compare_continuous_pos_not_zero) Prefer zero continuous positional.
preference(compare_continuous_pos_not_zero,M) :-
    attribute(task,root,(compare;relationship;distribution)),
    entity(view,root,_),
    attribute((view,coordinates),_,cartesian),
    entity(encoding,M,E1),
    entity(encoding,M,E2),
    attribute((encoding,channel),E1,x),
    attribute((encoding,channel),E2,y),
    attribute((scale,channel),S1,x),
    attribute((scale,channel),S2,y),
    not attribute((scale,zero),S1,true),
    attribute((scale,zero),S2,true),
    E1 != E2,
    S1 != S2.

% @soft(bin_datetime) Prefer to not bin datatime.
preference(bin_datetime,F) :-
    entity(field,_,F),
    entity(encoding,_,E),
    attribute((field,name),F,N),
    attribute((field,type),F,datetime),
    attribute((encoding,field),E,N),
    attribute((encoding,binning),E,_).

% @soft(compare_log_scale) log scale.
preference(compare_log_scale,E) :-
    attribute(task,root,compare),
    helper((encoding,scale_type),E,log).


% @soft(comparesum_ordinal) Prefer not to sum ordinal numbers.
preference(comparesum_ordinal,M) :-
    attribute(task,root,(relationship;distribution;compare)),
    entity(encoding,M,E),
    entity(scale,_,S),
    entity(field,_,F),
    attribute((field,name),F,N),
    attribute((field,type),F,number),
    attribute((encoding,field),E,N),
    attribute((encoding,channel),E,A),
    attribute((encoding,aggregate),E,(sum;median;mean;min;max;stdev;count)),
    attribute((scale,channel),S,A),
    attribute((scale,type),S,(ordinal;numcate)).

% @soft(compare_c_c_facet) Facet are not great in a continuous_continuous
preference(compare_c_c_facet,F) :-
    attribute(task,root,compare),
    entity(field,_,F1),
    entity(field,_,F2),
    attribute((field,name),F1,A),
    attribute((field,type),F1,number),
    attribute((field,name),F2,B),
    attribute((field,type),F2,number),
    entity(encoding,_,E1),
    entity(encoding,_,E2),
    attribute((encoding,field),E1,A),
    attribute((encoding,field),E2,B),
    attribute((encoding,channel),E1,x),
    attribute((encoding,channel),E2,y),
    attribute((facet,field),F,(A;B)).

% @soft(sum_categorical_number) Prefer not to sum categorical numbers.
preference(sum_categorical_number,M) :-
    attribute(task,root,compare),
    entity(encoding,M,E),
    entity(scale,_,S),
    entity(field,_,F),
    attribute((field,name),F,N),
    attribute((field,type),F,number),
    attribute((field,unique),F,U),
    attribute((encoding,field),E,N),
    attribute((encoding,channel),E,A),
    attribute((encoding,aggregate),E,(sum;median;mean;min;max;stdev)),
    attribute((scale,channel),S,A),
    U < 25.

% @soft(aggregate_string) Strings should not have aggregate functions
preference(aggregate_string,M) :-
    attribute(task,root,compare),
    entity(field,root,F),
    entity(encoding,M,E),
    attribute((field,name),F,N),
    attribute((field,type),F,string),
    attribute((encoding,field),E,N),
    attribute((encoding,aggregate),E,(max;min)).

% @soft(compare_facet_variables) Facets should not use already in use axial variables
preference(compare_facet_variables,M) :-
    attribute(task,root,(compare;relationship)),
    entity(field,root,F1),
    entity(field,root,F2),
    attribute((field,name),F1,N1),
    attribute((field,name),F2,N2),
    entity(encoding,M,E1),
    entity(encoding,M,E2),
    attribute((encoding,field),E1,N1),
    attribute((encoding,field),E2,N2),
    entity(facet,_,FA),
    attribute((facet,field),FA,(N1;N2)),
    F1 != F2,
    N1 != N2.

% @soft(compare_cd_overlap_line) Continuous by discrete for line mark in Compare tasks.
preference(compare_cd_overlap_line,M) :-
    attribute(task,root,compare),
    helper(is_c_d,M),
    not helper(no_overlap,M),
    attribute((mark,type),M,line).

% @soft(string_color_categorical) Strings and datetime should only be categorical.
preference(string_color_categorical,M) :-
    entity(field,_,F),
    entity(encoding,M,E),
    entity(scale,_,S),
    attribute((field,name),F,N),
    attribute((field,type),F,(string;datetime)),
    attribute((encoding,field),E,N),
    attribute((encoding,channel),E,color),
    not helper((encoding,scale_type),E,categorical).

% @soft(line_date_string) Line should not be used with datetime and string.
preference(line_date_string,M) :-
    attribute(task,root,relationship),
    entity(field,_,F1),
    entity(field,_,F2),
    entity(encoding,_,E1),
    entity(encoding,_,E2),
    attribute((field,name),F1,N1),
    attribute((field,type),F1,string),
    attribute((field,name),F2,N2),
    attribute((field,type),F2,datetime),
    attribute((encoding,field),E1,N1),
    attribute((encoding,field),E2,N2),
    attribute((encoding,channel),E1,(x;y)),
    attribute((encoding,channel),E2,(x;y)),
    attribute((mark,type),M,line).

% @soft(line_with_aggregate_distribution) Line should be used with aggregative fields in distribution.
preference(line_with_aggregate_distribution,M) :-
    attribute(task,root,(distribution;compare)),
    entity(field,_,F1),
    entity(field,_,F2),
    attribute((field,name),F1,N1),
    attribute((field,type),F1,(string;datetime)),
    attribute((field,name),F2,N2),
    attribute((field,type),F2,number),
    entity(encoding,_,E1),
    entity(encoding,_,E2),
    attribute((mark,type),M,line),
    attribute((encoding,field),E1,N1),
    attribute((encoding,channel),E1,(x;y)),
    attribute((encoding,field),E2,N2),
    attribute((encoding,channel),E2,(x;y)),
    not attribute((encoding,aggregate),_,_).

% @soft(line_with_aggregate_distribution) Line should be used with aggregative fields in distribution.
preference(line_with_aggregate_distribution,M) :-
    attribute(task,root,(distribution;compare)),
    entity(field,_,F1),
    entity(field,_,F2),
    attribute((field,name),F1,N1),
    attribute((field,type),F1,number),
    attribute((field,name),F2,N2),
    attribute((field,type),F2,number),
    entity(encoding,_,E1),
    entity(encoding,_,E2),
    attribute((mark,type),M,line),
    attribute((encoding,field),E1,N1),
    attribute((encoding,channel),E1,x),
    attribute((encoding,field),E2,N2),
    attribute((encoding,channel),E2,y),
    not attribute((encoding,aggregate),E2,_).

% @soft(aggre_linear) Aggregates should have no aggregates or all aggregates.
preference(aggre_linear,M) :-
    attribute(task,root,distribution),
    entity(field,_,F1),
    entity(field,_,F2),
    entity(encoding,M,E1),
    entity(encoding,M,E2),
    attribute((field,name),F1,N1),
    attribute((field,type),F1,number),
    attribute((field,name),F2,N2),
    attribute((field,type),F2,number),
    attribute((encoding,field),E1,N1),
    attribute((encoding,field),E2,N2),
    attribute((encoding,channel),E1,y),
    attribute((encoding,channel),E2,x),
    attribute((encoding,aggregate),E1,(sum;mean;median)).

% @soft(size_with_sum) Size is prefered to not have sum.
preference(size_with_sum,M) :-
    entity(field,_,F),
    entity(encoding,M,E),
    attribute((field,name),F,N),
    attribute((field,type),F,number),
    attribute((encoding,field),E,N),
    attribute((encoding,channel),E,size),
    attribute((encoding,aggregate),E,_).

% @soft(value_point) Point mark for value tasks.
preference(value_point,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,point).

% @soft(value_bar) Bar mark for value tasks.
preference(value_bar,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,bar).

% @soft(value_line) Line mark for value tasks.
preference(value_line,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,line).

% @soft(value_area) Area mark for value tasks.
preference(value_area,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,area).

% @soft(value_text) Text mark for value tasks.
preference(value_text,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,text).

% @soft(value_tick) Tick mark for value tasks.
preference(value_tick,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,tick).

% @soft(value_rect) Rect mark for value tasks.
preference(value_rect,M) :-
    attribute(task,root,value),
    attribute((mark,type),M,rect).

% @soft(summary_point) Point mark for summary tasks.
preference(summary_point,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,point).

% @soft(summary_bar) Bar mark for summary tasks.
preference(summary_bar,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,bar).

% @soft(summary_line) Line mark for summary tasks.
preference(summary_line,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,line).

% @soft(summary_area) Area mark for summary tasks.
preference(summary_area,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,area).

% @soft(summary_text) Text mark for summary tasks.
preference(summary_text,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,text).

% @soft(summary_tick) Tick mark for summary tasks.
preference(summary_tick,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,tick).

% @soft(summary_rect) Rect mark for summary tasks.
preference(summary_rect,M) :-
    attribute(task,root,summary),
    attribute((mark,type),M,rect).

% @soft(relationship_continuous_x) Continuous x for relationship tasks.
preference(relationship_continuous_x,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,x),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_continuous_y) Continuous y for relationship tasks.
preference(relationship_continuous_y,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,y),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_continuous_color) Continuous color for relationship tasks.
preference(relationship_continuous_color,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,color),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_continuous_size) Continuous size for relationship tasks.
preference(relationship_continuous_size,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,size),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_continuous_text) Continuous text for relationship tasks.
preference(relationship_continuous_text,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,text),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_x) Discrete x for relationship tasks.
preference(relationship_discrete_x,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,x),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_y) Discrete y for relationship tasks.
preference(relationship_discrete_y,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,y),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_color) Discrete color for relationship tasks.
preference(relationship_discrete_color,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,color),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_size) Discrete size for relationship tasks.
preference(relationship_discrete_size,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,size),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_shape) Discrete shape for relationship tasks.
preference(relationship_discrete_shape,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,shape),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_text) Discrete text for relationship tasks.
preference(relationship_discrete_text,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,text),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(relationship_discrete_detail) Discrete detail for relationship tasks.
preference(relationship_discrete_detail,E) :-
    attribute(task,root,relationship),
    attribute((encoding,channel),E,detail),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).


% @soft(distribution_ordinal_axis_facet) Prevents ordinal axis from being facet.
preference(distribution_ordinal_axis_facet,M) :-
    attribute(task,root,distribution),
    entity(field,_,A),
    entity(facet,_,F),
    entity(encoding,_,E),
    entity(scale,M,S),
    attribute((field,name),A,N),
    attribute((field,type),A,_),
    attribute((encoding,field),E,N),
    attribute((encoding,channel),E,(x;y)),
    attribute((scale,channel),S,C),
    attribute((scale,type),S,(ordinal;numcate)),
    attribute((facet,field),F,N).

% @soft(distribution_aggregate_ordinal) Prefer to not aggregate ordinal channels.
preference(distribution_aggregate_ordinal,E) :-
    attribute(task,root,distribution),
    entity(scale,_,S),
    attribute((scale,channel),S,C),
    attribute((scale,type),S,ordinal),
    attribute((encoding,field),E,_),
    attribute((encoding,channel),E,C),
    attribute((encoding,aggregate),E,_).


% @soft(distribution_discrete_size) Size for distribution tasks.
preference(distribution_discrete_size,E) :-
    attribute(task,root,distribution),
    attribute((encoding,channel),E,size),
    helper(mark_encoding_discrete_or_binned,_,E).

% @soft(distribution_continuous_size) Continuous size for distribution tasks.
preference(distribution_continuous_size,E) :-
    attribute(task,root,distribution),
    attribute((encoding,channel),E,size),
    not helper(mark_encoding_discrete_or_binned,_,E).

% @soft(numcate_no_aggregates) Numcate prefers to work without aggregates.
preference(numcate_no_aggregates,M) :-
    entity(encoding,M,E1),
    entity(encoding,M,E2),
    entity(scale,_,S1),
    entity(scale,_,S2),
    attribute((encoding,channel),E1,C),
    attribute((scale,channel),S1,C),
    attribute((scale,type),S1,numcate),
    attribute((encoding,channel),E2,_),
    attribute((encoding,aggregate),E2,_),
    E1 != E2.

% @soft(numcate_duplicates) Duplicas/Same field are not compatible with numcate.
preference(numcate_duplicates,M) :-
    entity(field,_,F),
    entity(encoding,M,E1),
    entity(encoding,M,E2),
    entity(scale,_,S),
    attribute((encoding,field),E1,F),
    attribute((encoding,channel),E1,(x;y)),
    attribute((scale,channel),S,(x;y)),
    attribute((scale,type),S,numcate),
    attribute((encoding,field),E2,F).

% @soft(value_continuous_x) Continuous x for value tasks.
preference(value_continuous_x,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,x),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_continuous_y) Continuous y for value tasks.
preference(value_continuous_y,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,y),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_continuous_color) Continuous color for value tasks.
preference(value_continuous_color,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,color),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_continuous_size) Continuous size for value tasks.
preference(value_continuous_size,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,size),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_continuous_text) Continuous text for value tasks.
preference(value_continuous_text,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,text),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_x) Discrete x for value tasks.
preference(value_discrete_x,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,x),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_y) Discrete y for value tasks.
preference(value_discrete_y,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,y),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_color) Discrete color for value tasks.
preference(value_discrete_color,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,color),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_size) Discrete size for value tasks.
preference(value_discrete_size,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,size),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_shape) Discrete shape for value tasks.
preference(value_discrete_shape,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,shape),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_text) Discrete text for value tasks.
preference(value_discrete_text,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,text),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(value_discrete_detail) Discrete detail for value tasks.
preference(value_discrete_detail,E) :-
    attribute(task,root,value),
    attribute((encoding,channel),E,detail),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_continuous_x) Continuous x for summary tasks.
preference(summary_continuous_x,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,x),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_continuous_y) Continuous y for summary tasks.
preference(summary_continuous_y,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,y),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_continuous_color) Continuous color for summary tasks.
preference(summary_continuous_color,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,color),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_continuous_size) Continuous size for summary tasks.
preference(summary_continuous_size,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,size),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_continuous_text) Continuous text for summary tasks.
preference(summary_continuous_text,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,text),
    not helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_x) Discrete x for summary tasks.
preference(summary_discrete_x,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,x),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_y) Discrete y for summary tasks.
preference(summary_discrete_y,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,y),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_color) Discrete color for summary tasks.
preference(summary_discrete_color,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,color),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_size) Discrete size for summary tasks.
preference(summary_discrete_size,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,size),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_shape) Discrete shape for summary tasks.
preference(summary_discrete_shape,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,shape),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_text) Discrete text for summary tasks.
preference(summary_discrete_text,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,text),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(summary_discrete_detail) Discrete detail for summary tasks.
preference(summary_discrete_detail,E) :-
    attribute(task,root,summary),
    attribute((encoding,channel),E,detail),
    helper(mark_encoding_discrete_or_binned,_,E),
    helper(enc_interesting,E).

% @soft(interesting_x) Interesting on x channel.
preference(interesting_x,E) :-
    attribute((encoding,channel),E,x),
    helper(enc_interesting,E).

% @soft(interesting_y) Interesting on y channel.
preference(interesting_y,E) :-
    attribute((encoding,channel),E,y),
    helper(enc_interesting,E).

% @soft(interesting_color) Interesting on color channel.
preference(interesting_color,E) :-
    attribute((encoding,channel),E,color),
    helper(enc_interesting,E).

% @soft(interesting_size) Interesting on size channel.
preference(interesting_size,E) :-
    attribute((encoding,channel),E,size),
    helper(enc_interesting,E).

% @soft(interesting_shape) Interesting on shape channel.
preference(interesting_shape,E) :-
    attribute((encoding,channel),E,shape),
    helper(enc_interesting,E).

% @soft(interesting_text) Interesting on text channel.
preference(interesting_text,E) :-
    attribute((encoding,channel),E,text),
    helper(enc_interesting,E).

% @soft(interesting_row) Interesting on row channel.
preference(interesting_row,E) :-
    attribute((facet,channel),E,row),
    helper(enc_interesting,E).

% @soft(interesting_column) Interesting on column channel.
preference(interesting_column,E) :-
    attribute((facet,channel),E,col),
    helper(enc_interesting,E).

% @soft(interesting_detail) Interesting on detail channel.
preference(interesting_detail,E) :-
    attribute((encoding,channel),E,detail),
    helper(enc_interesting,E).

% @soft(cartesian_coordinate) Cartesian coordinates.
preference(cartesian_coordinate,V) :-
    attribute((view,coordinates),V,cartesian).

% @soft(polar_coordinate) Polar coordinates.
preference(polar_coordinate,V) :-
    attribute((view,coordinates),V,polar).
