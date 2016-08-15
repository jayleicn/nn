local ParallelCriterion, parent = torch.class('nn.ParallelCriterion', 'nn.Criterion')

function ParallelCriterion:__init(repeatTarget)
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
   self.repeatTarget = repeatTarget
end

function ParallelCriterion:add(criterion, weight)
   assert(criterion, 'no criterion provided')
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

-- custom begins here, Jie Lei  
----------------------------------------------------------------------
-- indicator={scalar_1, scalar_2,..., scalar_num_of_criterion}
-- which indicates whether the i-th loss in the ParallelCriterion should be included for the specified training example


-- Example 
-- L1 = nn.CrossEntropyCriterion()
-- L2 = nn.CrossEntropyCriterion()
-- m = nn.ParallelCriterion():add(L1):add(L2)
-- m:forward({x_1, x_2}, {y_1, y_2}, {1, 1}) --with L = 1*L1 + 1*L2
-- m:forward({x_1, x_2}, {y_1, y_2}, {1, 0}) --with L = 1*L1 + 0*L2
-- m:backward({x_1, x_2}, {y_1, y_2}, {1, 0})

----------------------------------------------------------------------
-- this wiil be called by the forward function
function ParallelCriterion:updateOutput(input, target, indicator)
   self.output = 0
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      self.output = self.output + self.weights[i]*indicator[i]*criterion:updateOutput(input[i],target)
   end
   return self.output
end


-- This will be called by the backward function
function ParallelCriterion:updateGradInput(input, target, indicator)
   -- make sure gradInput has the same structure with the input,
   -- regardless it is a table contain Tensors of simply a Tensor.
   -- then initialize its value to zero
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i]*indicator[i], criterion:updateGradInput(input[i], target))
   end
   return self.gradInput
end

function ParallelCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end
